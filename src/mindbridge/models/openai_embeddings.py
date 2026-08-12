"""Jina recall embeddings through an OpenAI-compatible vLLM endpoint."""

from __future__ import annotations

from typing import Literal, TypedDict, cast

import openai
from openai import AsyncOpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse

from mindbridge.application.recall import RecallEmbeddingQuery
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


class OpenAIJinaTextEmbedder:
    """Batch Jina Text Small documents through the official OpenAI SDK."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model_reference: ModelReference,
        *,
        space_reference: EmbeddingSpaceReference = DEFAULT_JINA_RETRIEVAL_SPACE,
        dimension: int = DEFAULT_JINA_OMNI_DIMENSION,
        request_timeout_seconds: float = 120.0,
    ) -> None:
        if dimension <= 0 or request_timeout_seconds <= 0:
            raise ValueError("dimension and request timeout must be positive")
        self._client = client
        self._model_reference = model_reference
        self._space_reference = space_reference
        self._dimension = dimension
        self._request_timeout_seconds = request_timeout_seconds

    @classmethod
    def connect(
        cls,
        *,
        api_key: str,
        endpoint: str,
        model_id: str = DEFAULT_JINA_TEXT_MODEL_ID,
        model_revision: str = DEFAULT_JINA_TEXT_REVISION,
        space_reference: EmbeddingSpaceReference = DEFAULT_JINA_RETRIEVAL_SPACE,
        dimension: int = DEFAULT_JINA_OMNI_DIMENSION,
        request_timeout_seconds: float = 120.0,
        max_retries: int = 2,
    ) -> OpenAIJinaTextEmbedder:
        """Connect to one pinned OpenAI-compatible Text Small deployment."""
        if any(not value.strip() for value in (api_key, endpoint, model_id, model_revision)):
            raise ValueError(
                "text embedding credentials, endpoint, model ID, and revision are required"
            )
        if not 0 <= max_retries <= 10:
            raise ValueError("max_retries must be between zero and ten")
        return cls(
            AsyncOpenAI(
                api_key=api_key,
                base_url=normalize_openai_base_url(endpoint),
                timeout=request_timeout_seconds,
                max_retries=max_retries,
            ),
            ModelReference(model_id=model_id, revision=model_revision),
            space_reference=space_reference,
            dimension=dimension,
            request_timeout_seconds=request_timeout_seconds,
        )

    @property
    def model_reference(self) -> ModelReference:
        return self._model_reference

    @property
    def space_reference(self) -> EmbeddingSpaceReference:
        return self._space_reference

    @property
    def dimension(self) -> int:
        return self._dimension

    @trace_operation("mindbridge.model.encode_text_documents")
    async def encode_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        """Encode a bounded caller batch with Jina's document prompt semantics."""
        if not texts:
            return ()
        if any(not text.strip() for text in texts):
            raise DomainInvariantError("document texts must not be blank")
        set_current_span_attributes(
            {
                "mindbridge.model.id": self._model_reference.model_id,
                "mindbridge.model.revision": self._model_reference.revision,
                "mindbridge.embedding.dimension": self._dimension,
                "mindbridge.embedding.input_count": len(texts),
            }
        )
        try:
            response = await self._client.embeddings.create(
                input=[f"Document: {text}" for text in texts],
                model=self._model_reference.model_id,
                dimensions=self._dimension,
                encoding_format="float",
                timeout=self._request_timeout_seconds,
            )
        except openai.APIError as error:
            raise ModelUnavailableError("Jina embedding request failed") from error
        return _embedding_vectors(
            response,
            self._model_reference,
            dimension=self._dimension,
            expected_count=len(texts),
        )

    async def close(self) -> None:
        """Release connections owned by the OpenAI SDK client."""
        await self._client.close()


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
        self._document_embedder = OpenAIJinaTextEmbedder(
            document_client,
            document_model_reference,
            space_reference=space_reference,
            dimension=dimension,
            request_timeout_seconds=request_timeout_seconds,
        )
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
        return self._document_embedder.model_reference

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

        return _embedding_vectors(
            response,
            self._query_model_reference,
            dimension=self._dimension,
            expected_count=1,
        )[0]

    @trace_operation("mindbridge.model.encode_memory_document")
    async def encode_memory_document(self, text: str) -> tuple[float, ...]:
        """Encode one explicit memory with Jina's document-side prompt semantics."""
        return (await self._document_embedder.encode_documents((text,)))[0]

    async def close(self) -> None:
        """Release connections owned by the OpenAI SDK client."""
        await self._query_client.close()
        await self._document_embedder.close()

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


def _query_prompt(text: str) -> str:
    return f"Query: {text}"


def _embedding_vectors(
    response: CreateEmbeddingResponse,
    expected_model: ModelReference,
    *,
    dimension: int,
    expected_count: int,
) -> tuple[tuple[float, ...], ...]:
    if response.model != expected_model.model_id:
        raise ModelOutputError("embedding response model does not match the request")
    if len(response.data) != expected_count or {item.index for item in response.data} != set(
        range(expected_count)
    ):
        raise ModelOutputError("embedding response has invalid indices")
    vectors: list[tuple[float, ...]] = []
    for item in sorted(response.data, key=lambda value: value.index):
        if not isinstance(item.embedding, list):
            raise ModelOutputError("embedding response must use float encoding")
        vector = tuple(float(value) for value in item.embedding)
        validate_jina_embedding(vector, dimension)
        vectors.append(vector)
    return tuple(vectors)
