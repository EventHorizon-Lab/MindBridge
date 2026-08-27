"""Developer-facing local memory API."""

from __future__ import annotations

import asyncio
import base64
import builtins
import hashlib
import json
import math
import os
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Condition, RLock
from typing import Protocol, cast
from urllib.parse import urlsplit

from mindbridge.config import Config
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
    AssetDownloadError,
    AssetStore,
    AssetStoreError,
    AssetTooLargeError,
    UnsafeAssetUrlError,
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
    ModelBackend,
    ModelCapabilities,
    ModelInput,
    SpeechAnalysis,
    SpeechBackend,
)
from mindbridge.models.funasr import FunASRTranscriber
from mindbridge.models.jina import JinaOmniEmbedder
from mindbridge.models.openai_http import OpenAIHTTP
from mindbridge.types import (
    URL,
    AnswerResult,
    AssetRef,
    Blob,
    ContentAtom,
    ContentInput,
    MemoryRecord,
    Modality,
    Page,
    SearchHit,
    SpeakerSegment,
)

_DOCUMENT_TASK = EmbedTask.DOCUMENT.value
_INDEX_RECIPE = "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:single-vector-v2"
_OUTBOX_BATCH_SIZE = 256
_REINDEX_PAGE_SIZE = 256
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


@dataclass(slots=True)
class _OperationAssets:
    leased: builtins.list[StoredAsset]
    cleanup: builtins.list[StoredAsset]
    persisted: set[str]
    transcripts: dict[str, str]
    transcript_updates: dict[str, str]
    speech_updates: dict[str, SpeechAnalysis]


class Memory:
    """Persist and retrieve native text, image, video, audio, and omni memories."""

    def __init__(
        self,
        data_dir: str | Path = ".mindbridge",
        config: Config | None = None,
        *,
        models: ModelBackend | None = None,
        embedder: EmbeddingBackend | None = None,
        transcriber: SpeechBackend | None = None,
        speaker_similarity: float = 0.78,
        speaker_margin: float = 0.05,
    ) -> None:
        if config is not None and not isinstance(config, Config):
            raise ValidationError("config must be a Config value")
        self.config = config or Config.from_environment()
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

        self._store = _open_store(self.data_dir)
        try:
            self._models = _open_models(self.config, models)
            self._transcriber: SpeechBackend | ModelBackend = self._models
        except BaseException:
            self._store.close()
            raise
        try:
            self._embedder = _open_embedder(
                embedder,
                self._models,
                use_models=models is not None,
            )
        except BaseException:
            self._close_models_and_store(include_embedder=False)
            raise
        try:
            self._transcriber = _open_transcriber(
                transcriber,
                self._models,
                use_models=models is not None,
            )
        except BaseException:
            self._close_models_and_store()
            raise

        try:
            model_capabilities = _model_contract(self._models)
            transcription_capabilities, self._transcription_space = _transcription_contract(
                self._transcriber
            )
            if self._embedder is self._models:
                (
                    embedding_capabilities,
                    self._embedding_model,
                    self._space_id,
                    self._embedding_dimension,
                ) = _combined_embedding_contract(self._models, model_capabilities)
            else:
                (
                    embedding_capabilities,
                    self._embedding_model,
                    self._space_id,
                    self._embedding_dimension,
                ) = _embedding_contract(cast(EmbeddingBackend, self._embedder))
            self._capabilities = ModelCapabilities(
                embedding=embedding_capabilities,
                generation=model_capabilities.generation,
                transcription=transcription_capabilities,
            )
            self._assets = AssetStore(
                self.data_dir,
                allowed_url_hosts=self.config.allowed_url_hosts,
            )
            self._collect_orphan_assets(scan_physical=True)
            index_path = self.data_dir / "zvec"
            index_missing = not index_path.exists()
            self._ensure_store_metadata()
            if index_missing:
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
    ) -> MemoryRecord:
        """Add one native or mixed-modal memory and return its stable record."""
        with self._operation() as assets:
            prepared = _prepare_memory(
                self._prepare_content(content, assets),
                occurred_at=occurred_at,
                metadata=metadata,
            )
            return self._add_prepared((prepared,), operation=assets)[0]

    def add_many(self, contents: Sequence[ContentInput]) -> tuple[MemoryRecord, ...]:
        """Add memories in one model batch and one SQLite transaction."""
        with self._operation() as assets:
            if isinstance(contents, (str, bytes, Path, URL, Blob, AssetRef)):
                raise ValidationError("contents must be a sequence of memory inputs")
            prepared = tuple(
                _prepare_memory(
                    self._prepare_content(content, assets),
                    occurred_at=None,
                    metadata=None,
                )
                for content in contents
            )
            if not prepared:
                return ()
            return self._add_prepared(prepared, operation=assets)

    def search(self, query: ContentInput, *, limit: int = 10) -> tuple[SearchHit, ...]:
        """Return ranked memories for a native or mixed-modal query."""
        with self._operation() as assets:
            _limit(limit, maximum=100)
            prepared = self._prepare_content(query, assets)
            hits = self._search_prepared(
                prepared,
                limit=limit,
                operation=assets,
            )
            self._persist_transcripts(assets)
            return hits

    def ask(self, question: ContentInput, *, limit: int = 5) -> AnswerResult:
        """Answer a native or mixed-modal question only from retrieved memories."""
        with self._operation() as assets:
            _limit(limit, maximum=100)
            prepared = self._prepare_content(question, assets)
            hits = self._search_prepared(
                prepared,
                limit=limit,
                operation=assets,
            )
            routed_question = self._route_generation(prepared, assets)
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
            cached: dict[str, tuple[SpeakerSegment, ...]] = {}
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
                        cached[asset.asset_id] = segments
            if missing:
                analyses = self._analyze_speech(tuple(self._asset_ref(asset) for asset in missing))
                with self._write_lock, _translate_storage_errors("persist speaker recognition"):
                    for asset, analysis in zip(missing, analyses, strict=True):
                        cached[asset.asset_id] = self._store.write_speech(
                            asset.asset_id,
                            analysis,
                            model_id=self._transcriber.model_id,
                            space_id=self._transcription_space,
                            minimum_similarity=self._speaker_similarity,
                            minimum_margin=self._speaker_margin,
                        )
            return tuple(segment for asset in speech_assets for segment in cached[asset.asset_id])

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
            self._acknowledge_outbox()
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
            elif isinstance(atom, URL):
                name = atom.name or Path(urlsplit(atom.value).path).name or None
                modality, media_type = _media_hint(name, atom.media_type)
                candidate = self._assets.materialize_url(
                    atom.value,
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
        except (AssetDownloadError, AssetTooLargeError, UnsafeAssetUrlError, OSError, ValueError):
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
            if Modality.AUDIO not in self._capabilities.embedding:
                for memory in missing:
                    _fallback_unsupported(
                        memory.content,
                        self._capabilities.embedding,
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
        with self._write_lock:
            if stored_memories:
                candidate_ids = tuple(memory.memory_id for memory in stored_memories)
                with _translate_storage_errors("recheck existing memories"):
                    concurrent = self._store.read_memories(candidate_ids)
                concurrent_ids = {memory.memory_id for memory in concurrent}
                writes = tuple(
                    memory for memory in stored_memories if memory.memory_id not in concurrent_ids
                )
                embeddings = tuple(
                    embedding
                    for embedding in stored_embeddings
                    if embedding.memory_id not in concurrent_ids
                )
                if writes:
                    with _translate_storage_errors("write memories"):
                        self._store.write_memories(writes, embeddings)
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
    ) -> tuple[SearchHit, ...]:
        prepared = self._embedding_content(prepared, operation)
        model_input = self._route_embedding(prepared)
        vector = self._embed((model_input,), task=EmbedTask.QUERY)[0]
        with self._write_lock:
            self._drain_outbox()
            with _translate_index_errors("search memories"):
                if model_input.text:
                    index_hits = self._index.hybrid_search(
                        model_input.text,
                        vector,
                        limit=limit,
                        space_id=self._space_id,
                        task=_DOCUMENT_TASK,
                    )
                else:
                    index_hits = self._index.search(
                        vector,
                        limit=limit,
                        space_id=self._space_id,
                        task=_DOCUMENT_TASK,
                    )
            if not index_hits:
                return ()
            with _translate_storage_errors("hydrate search results"):
                memories = self._store.read_memories(tuple(hit.id for hit in index_hits))
            self._lease_assets(
                tuple(asset for memory in memories for asset in memory.assets),
                operation.leased,
            )
            operation.persisted.update(
                asset.asset_id for memory in memories for asset in memory.assets
            )
        relevance = {hit.id: hit.relevance for hit in index_hits}
        return tuple(self._search_hit(memory, relevance[memory.memory_id]) for memory in memories)

    def _route_embedding(self, prepared: _PreparedContent) -> ModelInput:
        value = self._resolved_model_input(prepared)
        missing = value.modalities - self._capabilities.embedding
        if not missing:
            return value
        if missing <= {Modality.AUDIO}:
            text = _derived_text(value.text, prepared.assets)
            assets = tuple(asset for asset in value.assets if asset.modality is not Modality.AUDIO)
            if text or assets:
                routed = ModelInput(text=text, assets=assets)
                missing = routed.modalities - self._capabilities.embedding
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
            self._capabilities.embedding,
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
            self._capabilities.generation,
            "generation",
        )
        if Modality.AUDIO in unsupported:
            prepared = self._with_audio_transcripts(prepared, operation)
        value = self._resolved_model_input(prepared)
        unsupported = value.modalities - self._capabilities.generation
        if not unsupported:
            return value
        text = _derived_text(value.text, prepared.assets)
        assets = tuple(asset for asset in value.assets if asset.modality is not Modality.AUDIO)
        if unsupported <= {Modality.AUDIO} and (text or assets):
            routed = ModelInput(
                text=text,
                assets=assets,
            )
            if not routed.modalities - self._capabilities.generation:
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
        if Modality.AUDIO not in self._capabilities.generation:
            for prepared in prepared_hits:
                _fallback_unsupported(
                    prepared,
                    self._capabilities.generation,
                    "generation",
                )
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
        text = _derived_text(prepared.text, assets)
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
            if Modality.AUDIO not in self._capabilities.transcription:
                raise ModelError(
                    "audio fallback requires a transcription model with audio capability"
                )
            by_id = {asset.asset_id: asset for asset in audio}
            refs = tuple(self._asset_ref(by_id[asset_id]) for asset_id in missing)
            try:
                if isinstance(self._transcriber, SpeechBackend):
                    analyses = self._analyze_speech(refs)
                    generated = tuple(
                        "\n".join(turn.text for turn in analysis.turns) for analysis in analyses
                    )
                    operation.speech_updates.update(zip(missing, analyses, strict=True))
                else:
                    generated = self._transcriber.transcribe(refs)
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
                            model_id=self._transcriber.model_id,
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
        try:
            result = self._models.answer(question, hits)
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

    def _acknowledge_outbox(self) -> None:
        """Acknowledge rebuild checkpoints after the full index is durable."""
        while True:
            with _translate_storage_errors("read the rebuilt search-index checkpoint"):
                operations = self._store.pending_index_operations(limit=_OUTBOX_BATCH_SIZE)
            if not operations:
                return
            with _translate_storage_errors("acknowledge the rebuilt search-index checkpoint"):
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

    def _ensure_store_metadata(self) -> None:
        expected = {
            _STORE_METADATA_KEYS["model"]: self._embedding_model,
            _STORE_METADATA_KEYS["space"]: self._space_id,
            _STORE_METADATA_KEYS["transcription"]: self._transcription_space,
            _STORE_METADATA_KEYS["dimension"]: str(self._embedding_dimension),
            _STORE_METADATA_KEYS["index"]: _INDEX_RECIPE,
        }
        with _translate_storage_errors("validate local store metadata"):
            for key, value in expected.items():
                stored = self._store.get_metadata(key)
                if stored is None:
                    self._store.set_metadata(key, value)
                elif stored != value:
                    raise StorageError(
                        f"local store metadata mismatch for {key}: expected {value!r}, "
                        f"found {stored!r}"
                    )

    def _close_models_and_store(self, *, include_embedder: bool = True) -> None:
        resources: tuple[_Closable, ...] = (self._transcriber, self._models, self._store)
        if include_embedder:
            resources = (self._embedder, *resources)
        for resource in _unique_resources(resources):
            with suppress(Exception):
                resource.close()

    def _close_resources(self) -> builtins.list[Exception]:
        failures: builtins.list[Exception] = []
        resources: tuple[_Closable, ...] = (
            self._embedder,
            self._transcriber,
            self._models,
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
        config: Config | None = None,
        *,
        models: ModelBackend | None = None,
        embedder: EmbeddingBackend | None = None,
        transcriber: SpeechBackend | None = None,
        speaker_similarity: float = 0.78,
        speaker_margin: float = 0.05,
    ) -> None:
        self._memory = Memory(
            data_dir=data_dir,
            config=config,
            models=models,
            embedder=embedder,
            transcriber=transcriber,
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
    ) -> MemoryRecord:
        return await asyncio.to_thread(
            self._memory.add,
            content,
            occurred_at=occurred_at,
            metadata=metadata,
        )

    async def add_many(
        self,
        contents: Sequence[ContentInput],
    ) -> tuple[MemoryRecord, ...]:
        return await asyncio.to_thread(self._memory.add_many, contents)

    async def search(self, query: ContentInput, *, limit: int = 10) -> tuple[SearchHit, ...]:
        return await asyncio.to_thread(self._memory.search, query, limit=limit)

    async def ask(self, question: ContentInput, *, limit: int = 5) -> AnswerResult:
        return await asyncio.to_thread(self._memory.ask, question, limit=limit)

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
    if isinstance(content, (str, Path, URL, Blob, AssetRef)):
        return (content,)
    if isinstance(content, bytes) or not isinstance(content, Sequence):
        raise ValidationError("content must be text, media, or an ordered sequence of them")
    atoms = tuple(content)
    if not atoms:
        raise ValidationError("content must not be empty")
    if len(atoms) > _MAX_CONTENT_PARTS:
        raise ValidationError(f"content must not exceed {_MAX_CONTENT_PARTS} parts")
    if any(not isinstance(atom, (str, Path, URL, Blob, AssetRef)) for atom in atoms):
        raise ValidationError("content contains an unsupported input value")
    return atoms


def _prepare_memory(
    content: _PreparedContent,
    *,
    occurred_at: datetime | None,
    metadata: Mapping[str, object] | None,
) -> _PreparedMemory:
    normalized_occurred_at = _occurred_at(occurred_at)
    metadata_json = _metadata_json(metadata)
    payload = json.dumps(
        {
            "parts": content.canonical_parts,
            "metadata": json.loads(metadata_json),
            "occurred_at": (
                None if normalized_occurred_at is None else _datetime_text(normalized_occurred_at)
            ),
        },
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


def _open_store(data_dir: Path) -> LocalStore:
    try:
        return LocalStore(data_dir)
    except (DataDirectoryInUseError, UnsupportedSchemaError) as error:
        raise StorageError(str(error)) from error
    except Exception as error:
        raise StorageError("failed to open the local memory store") from error


def _open_models(config: Config, models: ModelBackend | None) -> ModelBackend:
    try:
        return OpenAIHTTP(config) if models is None else models
    except MindBridgeError:
        raise
    except Exception as error:
        raise ModelError("failed to open the configured models") from error


def _open_embedder(
    embedder: EmbeddingBackend | None,
    models: ModelBackend,
    *,
    use_models: bool,
) -> EmbeddingBackend | ModelBackend:
    try:
        if embedder is not None:
            return embedder
        return models if use_models else JinaOmniEmbedder()
    except MindBridgeError:
        raise
    except Exception as error:
        raise ModelError("failed to open the embedding model") from error


def _open_transcriber(
    transcriber: SpeechBackend | None,
    models: ModelBackend,
    *,
    use_models: bool,
) -> SpeechBackend | ModelBackend:
    try:
        if transcriber is not None:
            return transcriber
        return models if use_models else FunASRTranscriber()
    except MindBridgeError:
        raise
    except Exception as error:
        raise ModelError("failed to open the speech model") from error


def _model_contract(models: ModelBackend) -> ModelCapabilities:
    capabilities = models.capabilities
    if not isinstance(capabilities, ModelCapabilities):
        raise ValidationError("models.capabilities must be a ModelCapabilities value")
    return capabilities


def _transcription_contract(
    transcriber: SpeechBackend | ModelBackend,
) -> tuple[frozenset[Modality], str]:
    if isinstance(transcriber, SpeechBackend):
        capabilities = transcriber.capabilities
        if not isinstance(capabilities, frozenset):
            raise ValidationError("transcriber.capabilities must be a frozenset")
        normalized = ModelCapabilities(
            embedding=frozenset(),
            generation=frozenset(),
            transcription=capabilities,
        ).transcription
        return normalized, _model_text(transcriber.space_id, "transcription space")
    model_capabilities = _model_contract(transcriber)
    return model_capabilities.transcription, _model_text(
        transcriber.transcription_space,
        "transcription space",
    )


def _combined_embedding_contract(
    models: ModelBackend,
    capabilities: ModelCapabilities,
) -> tuple[frozenset[Modality], str, str, int]:
    space = _embedding_space(models.embedding_space)
    return (
        capabilities.embedding,
        _model_text(models.embedding_model, "embedding model"),
        space,
        _positive_dimension(models.embedding_dimension, "models.embedding_dimension"),
    )


def _embedding_contract(
    embedder: EmbeddingBackend,
) -> tuple[frozenset[Modality], str, str, int]:
    capabilities = embedder.capabilities
    if not isinstance(capabilities, frozenset) or any(
        not isinstance(modality, Modality) for modality in capabilities
    ):
        raise ValidationError("embedder.capabilities must be a frozenset of modalities")
    normalized = ModelCapabilities(
        embedding=capabilities,
        generation=frozenset(),
        transcription=frozenset(),
    ).embedding
    return (
        normalized,
        _model_text(embedder.model_id, "embedding model"),
        _embedding_space(embedder.space_id),
        _positive_dimension(embedder.dimension, "embedder.dimension"),
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
