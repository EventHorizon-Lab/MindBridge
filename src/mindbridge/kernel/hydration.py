"""Hydrating stored rows into the public record and hit values."""

from __future__ import annotations

from mindbridge.control import load_operation
from mindbridge.exceptions import StorageError
from mindbridge.infrastructure.local.store import StoredAsset, StoredMemory, StoredOperation
from mindbridge.kernel.content import PreparedContent, decode_metadata
from mindbridge.kernel.runtime import Storage, translate_storage_errors
from mindbridge.models.base import ModelInput
from mindbridge.types import (
    AssetRef,
    MemoryOperationRecord,
    MemoryOutcome,
    MemoryRecord,
    MemoryTrigger,
    MemoryType,
    Modality,
    SearchHit,
)


def operation_record(logged: StoredOperation) -> MemoryOperationRecord:
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
        superseded=logged.superseded,
        rolled_back_at=logged.rolled_back_at,
        outcome=None if logged.outcome is None else _memory_outcome(logged.outcome),
        outcome_note=logged.outcome_note,
    )


def _memory_outcome(value: str) -> MemoryOutcome:
    try:
        return MemoryOutcome(value)
    except ValueError:
        raise StorageError("a logged memory operation has an unknown outcome") from None


class Hydrator:
    """Turn stored rows into public `MemoryRecord`, `SearchHit`, and `AssetRef` values."""

    def __init__(
        self,
        *,
        storage: Storage,
    ) -> None:
        self._assets = storage.assets

    def memory_record(self, memory: StoredMemory) -> MemoryRecord:
        return MemoryRecord(
            id=memory.memory_id,
            content=memory.content,
            created_at=memory.created_at,
            occurred_at=memory.occurred_at,
            occurred_end=memory.occurred_end,
            metadata=decode_metadata(memory.metadata_json),
            assets=tuple(self.asset_ref(asset) for asset in memory.assets),
            modality=Modality(memory.modality),
            memory_type=MemoryType(memory.memory_type),
            context=memory.context,
            forgotten_at=memory.forgotten_at,
            place_id=memory.place_id,
        )

    def search_hit(self, memory: StoredMemory, relevance: float) -> SearchHit:
        return SearchHit(
            id=memory.memory_id,
            content=memory.content,
            score=max(0.0, min(1.0, relevance)),
            created_at=memory.created_at,
            occurred_at=memory.occurred_at,
            occurred_end=memory.occurred_end,
            metadata=decode_metadata(memory.metadata_json),
            assets=tuple(self.asset_ref(asset) for asset in memory.assets),
            modality=Modality(memory.modality),
            memory_type=MemoryType(memory.memory_type),
            context=memory.context,
            forgotten_at=memory.forgotten_at,
            place_id=memory.place_id,
        )

    def asset_ref(self, asset: StoredAsset) -> AssetRef:
        with translate_storage_errors("resolve local media"):
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

    def model_input(self, prepared: PreparedContent) -> ModelInput:
        return ModelInput(
            text=prepared.text,
            assets=tuple(self.asset_ref(asset) for asset in prepared.assets),
        )
