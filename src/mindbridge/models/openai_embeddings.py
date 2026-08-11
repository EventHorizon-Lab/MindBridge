"""Jina recall embeddings through an OpenAI-compatible vLLM endpoint."""

from __future__ import annotations

from typing import Literal, TypedDict, cast

import openai
from openai import AsyncOpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse

from mindbridge.application import RecallEmbeddingQuery
from mindbridge.core import (
    DomainInvariantError,
    EmbeddingSpaceReference,
    ModelOutputError,
    ModelReference,
    ModelUnavailableError,
)
from mindbridge.models.jina import (
    DEFAULT_JINA_OMNI_DIMENSION,
    DEFAULT_JINA_RETRIEVAL_SPACE,
    DEFAULT_JINA_TEXT_MODEL_ID,
    DEFAULT_JINA_TEXT_REVISION,
    validate_jina_embedding,
)
from mindbridge.models.openai_media import OpenAIContentPart, media_url_content_part
from mindbridge.models.openai_omni import (
    DEFAULT_VIDEO_FRAMES_PER_SECOND,
    DEFAULT_VIDEO_MAX_PIXELS,
    normalize_openai_base_url,
)
from mindbridge.telemetry import set_current_span_attributes, trace_operation


class _UserMessage(TypedDict):
    role: Literal["user"]
    content: list[OpenAIContentPart]


class OpenAIJinaEmbedder:
    """Encode queries and memory documents without loading Jina in the API."""

    def __init__(
        self,
        query_client: AsyncOpenAI,
        query_model_reference: ModelReference,
        *,
        document_client: AsyncOpenAI,
        document_model_reference: ModelReference,
        space_reference: EmbeddingSpaceReference = DEFAULT_JINA_RETRIEVAL_SPACE,
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
        self._query_client = query_client
        self._query_model_reference = query_model_reference
        self._document_client = document_client
        self._document_model_reference = document_model_reference
        self._space_reference = space_reference
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
        document_api_key: str,
        document_endpoint: str,
        document_model_id: str = DEFAULT_JINA_TEXT_MODEL_ID,
        document_model_revision: str = DEFAULT_JINA_TEXT_REVISION,
        space_reference: EmbeddingSpaceReference = DEFAULT_JINA_RETRIEVAL_SPACE,
        dimension: int = DEFAULT_JINA_OMNI_DIMENSION,
        request_timeout_seconds: float = 120.0,
        max_retries: int = 2,
    ) -> OpenAIJinaEmbedder:
        """Create the official SDK client for a pinned vLLM deployment."""
        required_values = (
            api_key,
            endpoint,
            model_id,
            model_revision,
            document_api_key,
            document_endpoint,
            document_model_id,
            document_model_revision,
        )
        if any(not value.strip() for value in required_values):
            raise ValueError(
                "embedding credentials, endpoints, model IDs, and revisions are required"
            )
        if not 0 <= max_retries <= 10:
            raise ValueError("max_retries must be between zero and ten")
        query_client = AsyncOpenAI(
            api_key=api_key,
            base_url=normalize_openai_base_url(endpoint),
            timeout=request_timeout_seconds,
            max_retries=max_retries,
        )
        return cls(
            query_client,
            ModelReference(model_id=model_id, revision=model_revision),
            document_client=AsyncOpenAI(
                api_key=document_api_key,
                base_url=normalize_openai_base_url(document_endpoint),
                timeout=request_timeout_seconds,
                max_retries=max_retries,
            ),
            document_model_reference=ModelReference(
                model_id=document_model_id,
                revision=document_model_revision,
            ),
            space_reference=space_reference,
            dimension=dimension,
            request_timeout_seconds=request_timeout_seconds,
        )

    @property
    def query_model_reference(self) -> ModelReference:
        return self._query_model_reference

    @property
    def document_model_reference(self) -> ModelReference:
        return self._document_model_reference

    @property
    def space_reference(self) -> EmbeddingSpaceReference:
        return self._space_reference

    @property
    def dimension(self) -> int:
        return self._dimension

    @trace_operation("mindbridge.model.encode_query")
    async def encode_query(self, query: RecallEmbeddingQuery) -> tuple[float, ...]:
        """Encode one retrieval query with Jina's query-side prompt semantics."""
        set_current_span_attributes(
            {
                "mindbridge.model.id": self._query_model_reference.model_id,
                "mindbridge.model.revision": self._query_model_reference.revision,
                "mindbridge.embedding.dimension": self._dimension,
                "mindbridge.query.media_count": len(query.media),
                "mindbridge.query.has_text": query.text is not None,
            }
        )
        try:
            if query.media:
                response = await self._encode_multimodal(query)
            else:
                response = await self._query_client.embeddings.create(
                    input=[_query_prompt(cast(str, query.text))],
                    model=self._query_model_reference.model_id,
                    dimensions=self._dimension,
                    encoding_format="float",
                    timeout=self._request_timeout_seconds,
                )
        except openai.APIError as error:
            raise ModelUnavailableError("Jina embedding request failed") from error

        return self._embedding_vector(response, self._query_model_reference)

    @trace_operation("mindbridge.model.encode_memory_document")
    async def encode_memory_document(self, text: str) -> tuple[float, ...]:
        """Encode one explicit memory with Jina's document-side prompt semantics."""
        set_current_span_attributes(
            {
                "mindbridge.model.id": self._document_model_reference.model_id,
                "mindbridge.model.revision": self._document_model_reference.revision,
                "mindbridge.embedding.dimension": self._dimension,
            }
        )
        if not text.strip():
            raise DomainInvariantError("memory document text must not be blank")
        try:
            response = await self._document_client.embeddings.create(
                input=[f"Document: {text}"],
                model=self._document_model_reference.model_id,
                dimensions=self._dimension,
                encoding_format="float",
                timeout=self._request_timeout_seconds,
            )
        except openai.APIError as error:
            raise ModelUnavailableError("Jina embedding request failed") from error
        return self._embedding_vector(response, self._document_model_reference)

    async def close(self) -> None:
        """Release connections owned by the OpenAI SDK client."""
        await self._query_client.close()
        await self._document_client.close()

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
        return await self._query_client.post(
            "/embeddings",
            cast_to=CreateEmbeddingResponse,
            body={
                "messages": messages,
                "model": self._query_model_reference.model_id,
                "dimensions": self._dimension,
                "encoding_format": "float",
            },
            options={"timeout": self._request_timeout_seconds},
        )

    def _embedding_vector(
        self,
        response: CreateEmbeddingResponse,
        expected_model: ModelReference,
    ) -> tuple[float, ...]:
        if response.model != expected_model.model_id:
            raise ModelOutputError("embedding response model does not match the request")
        if len(response.data) != 1 or response.data[0].index != 0:
            raise ModelOutputError("embedding response must contain one indexed vector")
        values = response.data[0].embedding
        if not isinstance(values, list):
            raise ModelOutputError("embedding response must use float encoding")
        vector = tuple(float(value) for value in values)
        validate_jina_embedding(vector, self._dimension)
        return vector


def _query_prompt(text: str) -> str:
    return f"Query: {text}"
