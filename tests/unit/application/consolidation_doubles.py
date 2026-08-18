"""Doubles shared byte-for-byte by the three consolidation use-case checks."""

from __future__ import annotations

from datetime import datetime, timedelta

from mindbridge.application.capabilities import (
    Embedding,
    EmbedRequest,
    EmbedResult,
    TextPart,
)
from mindbridge.application.ports import PresignedMediaDownload
from mindbridge.core import EmbeddingSpaceReference, MediaObject, ModelReference


class RecordingTextEmbedder:
    """Returns a valid unit vector while retaining the text that was embedded."""

    space_reference = EmbeddingSpaceReference(space_id="jina-v5", revision="space-v1")

    def __init__(self) -> None:
        self.documents: tuple[str, ...] = ()

    async def embed(self, request: EmbedRequest) -> EmbedResult:
        self.documents = tuple(
            part.text
            for input_value in request.inputs
            for part in input_value.parts
            if isinstance(part, TextPart)
        )
        return EmbedResult(
            tuple(
                Embedding(
                    (1.0, 0.0),
                    ModelReference(model_id="jina-text", revision="text-revision"),
                    EmbeddingSpaceReference(space_id="jina-v5", revision="space-v1"),
                )
                for _ in request.inputs
            )
        )


class DeterministicSigner:
    """Keeps evidence resolution independent from object storage.

    The signing instant is passed in so the URL expiry always tracks the calling test's own
    clock rather than a second one that could drift away from it unnoticed.
    """

    def __init__(self, signed_at: datetime) -> None:
        self._signed_at = signed_at

    async def create_presigned_download(
        self,
        media_object: MediaObject,
    ) -> PresignedMediaDownload:
        return PresignedMediaDownload(
            download_url=f"https://objects.example.test/{media_object.media_object_id}",
            expires_at=self._signed_at + timedelta(minutes=5),
        )
