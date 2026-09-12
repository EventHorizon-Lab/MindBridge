"""Materializing caller content: validating atoms and leasing media before any model runs."""

from __future__ import annotations

import builtins
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from mindbridge.exceptions import MindBridgeError, StorageError, ValidationError
from mindbridge.infrastructure.local.assets import AssetStoreError, AssetTooLargeError
from mindbridge.infrastructure.local.store import StoredAsset
from mindbridge.kernel.content import (
    PreparedContent,
    PreparedMemory,
    content_atoms,
    media_hint,
    memory_modality,
    prepare_memory,
)
from mindbridge.kernel.lifecycle import Lifecycle, OperationAssets
from mindbridge.kernel.runtime import Storage, translate_storage_errors
from mindbridge.kernel.validation import MAX_TEXT_CHARACTERS, validated_text
from mindbridge.types import (
    AssetRef,
    Blob,
    ContentAtom,
    ContentInput,
    MemoryType,
    ObservationContext,
)


class Materializer:
    """Validate caller content and lease its media before any model receives it."""

    def __init__(
        self,
        *,
        storage: Storage,
        lifecycle: Lifecycle,
    ) -> None:
        self._assets = storage.assets
        self._store = storage.store
        self._write_lock = storage.write_lock
        self._lifecycle = lifecycle

    def prepare(
        self,
        content: ContentInput,
        operation: OperationAssets,
    ) -> PreparedContent:
        atoms = content_atoms(content)
        text_parts: builtins.list[str] = []
        assets: builtins.list[StoredAsset] = []
        canonical: builtins.list[tuple[str, str]] = []
        for atom in atoms:
            if isinstance(atom, str):
                text = validated_text(atom, "content")
                text_parts.append(text)
                canonical.append(("text", text))
                continue
            asset = self._materialize_atom(atom, operation)
            assets.append(asset)
            canonical.append(("asset", asset.sha256))
        text = "\n\n".join(text_parts)
        if len(text) > MAX_TEXT_CHARACTERS:
            raise ValidationError(f"content text must not exceed {MAX_TEXT_CHARACTERS} characters")
        return PreparedContent(
            text=text,
            assets=tuple(assets),
            modality=memory_modality(assets),
            canonical_parts=tuple(canonical),
        )

    def prepare_batch_item(
        self,
        content: ContentInput,
        *,
        index: int,
        operation: OperationAssets,
        occurred_at: datetime | None,
        occurred_end: datetime | None,
        metadata: Mapping[str, object] | None,
        memory_type: MemoryType,
        context: ObservationContext | None,
    ) -> PreparedMemory:
        """Prepare one batch item, naming its position when it is the item that fails."""
        try:
            return prepare_memory(
                self.prepare(content, operation),
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

    def _materialize_atom(
        self,
        atom: ContentAtom,
        operation: OperationAssets,
    ) -> StoredAsset:
        try:
            if isinstance(atom, Path):
                modality, media_type = media_hint(atom.name, None)
                candidate = self._assets.materialize_path(
                    atom,
                    modality=modality.value,
                    mime_type=media_type,
                    lease=True,
                )
            elif isinstance(atom, Blob):
                modality, media_type = media_hint(atom.name, atom.media_type)
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
        with translate_storage_errors("resolve media metadata"):
            stored = self._store.media.read_asset(candidate.asset_id)
        if stored is not None:
            self._validate_asset_match(candidate, stored)
            operation.persisted.add(stored.asset_id)
            return stored
        return candidate

    def _resolve_asset_reference(
        self,
        reference: AssetRef,
        operation: OperationAssets,
    ) -> StoredAsset:
        try:
            with self._write_lock, translate_storage_errors("resolve media reference"):
                asset = self._store.media.read_asset(reference.id)
                if asset is not None:
                    self._lifecycle.lease_assets((asset,), operation.leased)
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
