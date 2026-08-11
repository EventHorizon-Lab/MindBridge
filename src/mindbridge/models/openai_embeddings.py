"""Jina recall-query embeddings through an OpenAI-compatible vLLM endpoint."""

from __future__ import annotations

from typing import Literal, TypedDict, cast

import openai
from openai import AsyncOpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse

from mindbridge.application import RecallEmbeddingQuery
from mindbridge.core import ModelOutputError, ModelReference, ModelUnavailableError
from mindbridge.models.jina import DEFAULT_JINA_OMNI_DIMENSION, validate_jina_embedding
from mindbridge.models.openai_media import OpenAIContentPart, media_url_content_part
from mindbridge.models.openai_omni import (
    DEFAULT_VIDEO_FRAMES_PER_SECOND,
    DEFAULT_VIDEO_MAX_PIXELS,
    normalize_openai_base_url,
)


class _UserMessage(TypedDict):
    role: Literal["user"]
    content: list[OpenAIContentPart]


class OpenAIJinaQueryEmbedder:
    """Encode fused text and AV retrieval queries without loading Jina in the API."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model_reference: ModelReference,
        *,
        dimension: int = DEFAULT_JINA_OMNI_DIMENSION,
        request_timeout_seconds: float = 120.0,
        video_frames_per_second: float = DEFAULT_VIDEO_FRAMES_PER_SECOND,
        video_max_pixels: int = DEFAULT_VIDEO_MAX_PIXELS,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if request_timeout_seconds <= 0:
            raise ValueError("request timeout must be positive")
        if video_frames_per_second <= 0 or video_max_pixels <= 0:
            raise ValueError("video sampling values must be positive")
        self._client = client
        self._model_reference = model_reference
        self._dimension = dimension
        self._request_timeout_seconds = request_timeout_seconds
        self._video_frames_per_second = video_frames_per_second
        self._video_max_pixels = video_max_pixels

    @classmethod
    def connect(
        cls,
        *,
        api_key: str,
        endpoint: str,
        model_id: str,
        model_revision: str,
        dimension: int = DEFAULT_JINA_OMNI_DIMENSION,
        request_timeout_seconds: float = 120.0,
        max_retries: int = 2,
    ) -> OpenAIJinaQueryEmbedder:
        """Create the official SDK client for a pinned vLLM deployment."""
        if not api_key.strip() or not model_id.strip() or not model_revision.strip():
            raise ValueError("embedding API key, model ID, and revision must not be empty")
        if not 0 <= max_retries <= 10:
            raise ValueError("max_retries must be between zero and ten")
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=normalize_openai_base_url(endpoint),
            timeout=request_timeout_seconds,
            max_retries=max_retries,
        )
        return cls(
            client,
            ModelReference(model_id=model_id, revision=model_revision),
            dimension=dimension,
            request_timeout_seconds=request_timeout_seconds,
        )

    @property
    def model_reference(self) -> ModelReference:
        return self._model_reference

    @property
    def dimension(self) -> int:
        return self._dimension

    async def encode_query(self, query: RecallEmbeddingQuery) -> tuple[float, ...]:
        """Encode one retrieval query with Jina's query-side prompt semantics."""
        try:
            if query.media:
                response = await self._encode_multimodal(query)
            else:
                response = await self._client.embeddings.create(
                    input=[_query_prompt(cast(str, query.text))],
                    model=self._model_reference.model_id,
                    dimensions=self._dimension,
                    encoding_format="float",
                    timeout=self._request_timeout_seconds,
                )
        except openai.APIError as error:
            raise ModelUnavailableError("Jina embedding request failed") from error

        if response.model != self._model_reference.model_id:
            raise ModelOutputError("embedding response model does not match the request")
        if len(response.data) != 1 or response.data[0].index != 0:
            raise ModelOutputError("embedding response must contain one indexed vector")
        values = response.data[0].embedding
        if not isinstance(values, list):
            raise ModelOutputError("embedding response must use float encoding")
        vector = tuple(float(value) for value in values)
        validate_jina_embedding(vector, self._dimension)
        return vector

    async def close(self) -> None:
        """Release connections owned by the OpenAI SDK client."""
        await self._client.close()

    async def _encode_multimodal(
        self,
        query: RecallEmbeddingQuery,
    ) -> CreateEmbeddingResponse:
        content: list[OpenAIContentPart] = [
            {"type": "text", "text": _query_prompt(query.text or "")}
        ]
        content.extend(
            media_url_content_part(
                item.media_object.kind,
                item.media_url,
                source_uri=item.media_object.uri,
                video_frames_per_second=self._video_frames_per_second,
                video_max_pixels=self._video_max_pixels,
            )
            for item in query.media
        )
        messages: list[_UserMessage] = [{"role": "user", "content": content}]
        return await self._client.post(
            "/embeddings",
            cast_to=CreateEmbeddingResponse,
            body={
                "messages": messages,
                "model": self._model_reference.model_id,
                "dimensions": self._dimension,
                "encoding_format": "float",
            },
            options={"timeout": self._request_timeout_seconds},
        )


def _query_prompt(text: str) -> str:
    return f"Query: {text}"
