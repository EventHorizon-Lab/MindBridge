"""Typed inputs and boundaries for multimodal candidate recall."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from mindbridge.core import DomainInvariantError, MediaObject, ModelReference

RETRIEVAL_DOCUMENT_EMBEDDING_TASK = "retrieval_document"


@dataclass(frozen=True, slots=True)
class ResolvedQueryMedia:
    """One tenant-owned query object with short-lived model access."""

    media_object: MediaObject
    media_url: str
    media_url_expires_at: datetime

    def __post_init__(self) -> None:
        if not self.media_url.strip():
            raise DomainInvariantError("query media URL must not be empty")
        if self.media_url_expires_at.utcoffset() is None:
            raise DomainInvariantError("query media URL expiry must be timezone-aware")


@dataclass(frozen=True, slots=True)
class RecallEmbeddingQuery:
    """Text and original AV fused into one retrieval-side embedding input."""

    text: str | None
    media: tuple[ResolvedQueryMedia, ...]

    def __post_init__(self) -> None:
        if self.text is not None and not self.text.strip():
            raise DomainInvariantError("embedding query text must not be blank")
        if self.text is None and not self.media:
            raise DomainInvariantError("embedding query requires text or media")
        media_ids = [item.media_object.media_object_id for item in self.media]
        if len(set(media_ids)) != len(media_ids):
            raise DomainInvariantError("embedding query media IDs must be unique")
        if len({item.media_object.tenant_id for item in self.media}) > 1:
            raise DomainInvariantError("embedding query media must belong to one tenant")


class RecallEmbedder(Protocol):
    """Frozen encoder shared by recall queries and explicit memory documents."""

    @property
    def model_reference(self) -> ModelReference: ...

    @property
    def dimension(self) -> int: ...

    async def encode_query(self, query: RecallEmbeddingQuery) -> tuple[float, ...]: ...

    async def encode_memory_document(self, text: str) -> tuple[float, ...]: ...
