"""OpenAI-compatible adapters for MindBridge's atomic model capabilities."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import cast
from urllib.parse import urlsplit, urlunsplit

import openai
from openai import AsyncOpenAI, Omit, omit
from openai.types.chat import ChatCompletionMessageParam
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.shared import ReasoningEffort
from openai.types.shared_params import ResponseFormatJSONObject

from mindbridge.application.capabilities import (
    Embedding,
    EmbedRequest,
    EmbedResult,
    EmbedTask,
    GenerateRequest,
    GenerateResult,
    ModelInput,
    TextPart,
)
from mindbridge.core import (
    EmbeddingSpaceReference,
    MediaKind,
    ModelOutputError,
    ModelReference,
    ModelRequestError,
    ModelUnavailableError,
)
from mindbridge.models.defaults import DEFAULT_GENERATOR_MODEL_ID
from mindbridge.telemetry import set_current_span_attributes, trace_operation

DEFAULT_VIDEO_FRAMES_PER_SECOND = 1.0
DEFAULT_VIDEO_MAX_PIXELS = 200_704
REASONING_EFFORT_VALUES = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


class OpenAIGenerator:
    """Generate text through the official SDK or a compatible endpoint."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model_reference: ModelReference,
        *,
        request_timeout_seconds: float = 1_800.0,
        reasoning_effort: ReasoningEffort = None,
        video_frames_per_second: float = DEFAULT_VIDEO_FRAMES_PER_SECOND,
        video_max_pixels: int = DEFAULT_VIDEO_MAX_PIXELS,
    ) -> None:
        if request_timeout_seconds <= 0:
            raise ValueError("request timeout must be positive")
        if video_frames_per_second <= 0 or video_max_pixels <= 0:
            raise ValueError("video sampling values must be positive")
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORT_VALUES:
            raise ValueError("reasoning effort is not supported")
        self._client = client
        self._model_reference = model_reference
        self._request_timeout_seconds = request_timeout_seconds
        self._reasoning_effort = reasoning_effort
        self._video_frames_per_second = video_frames_per_second
        self._video_max_pixels = video_max_pixels

    @classmethod
    def connect(
        cls,
        *,
        api_key: str,
        endpoint: str,
        model_id: str = DEFAULT_GENERATOR_MODEL_ID,
        model_revision: str,
        request_timeout_seconds: float = 1_800.0,
        max_retries: int = 2,
        reasoning_effort: ReasoningEffort = None,
        video_frames_per_second: float = DEFAULT_VIDEO_FRAMES_PER_SECOND,
        video_max_pixels: int = DEFAULT_VIDEO_MAX_PIXELS,
    ) -> OpenAIGenerator:
        if any(not value.strip() for value in (api_key, model_id, model_revision)):
            raise ValueError("API key, model ID, and model revision are required")
        if not 0 <= max_retries <= 10:
            raise ValueError("max_retries must be between zero and ten")
        return cls(
            AsyncOpenAI(
                api_key=api_key,
                base_url=normalize_base_url(endpoint),
                timeout=request_timeout_seconds,
                max_retries=max_retries,
            ),
            ModelReference(model_id=model_id, revision=model_revision),
            request_timeout_seconds=request_timeout_seconds,
            reasoning_effort=reasoning_effort,
            video_frames_per_second=video_frames_per_second,
            video_max_pixels=video_max_pixels,
        )

    @trace_operation("mindbridge.model.generate")
    async def generate(self, request: GenerateRequest) -> GenerateResult:
        """Stream one deterministic text result and normalize provider failures."""
        set_current_span_attributes(
            {
                "mindbridge.model.id": self._model_reference.model_id,
                "mindbridge.model.revision": self._model_reference.revision,
                "mindbridge.model.input_part_count": len(request.input.parts),
            }
        )
        response_format: ResponseFormatJSONObject | Omit = (
            {"type": "json_object"} if request.json_mode else omit
        )
        messages = cast(
            list[ChatCompletionMessageParam],
            [
                {"role": "system", "content": request.system_prompt},
                {
                    "role": "user",
                    "content": _content_parts(
                        request.input,
                        for_embedding=False,
                        video_frames_per_second=self._video_frames_per_second,
                        video_max_pixels=self._video_max_pixels,
                    ),
                },
            ],
        )
        parts: list[str] = []
        finish_reason: str | None = None
        system_fingerprint: str | None = None
        try:
            stream = await self._client.chat.completions.create(
                model=self._model_reference.model_id,
                messages=messages,
                modalities=["text"],
                max_tokens=request.max_output_tokens,
                response_format=response_format,
                reasoning_effort=(
                    self._reasoning_effort if self._reasoning_effort is not None else omit
                ),
                temperature=0.0,
                stream=True,
                stream_options={"include_usage": True},
                timeout=self._request_timeout_seconds,
            )
            async for chunk in stream:
                system_fingerprint = system_fingerprint or chunk.system_fingerprint
                for choice in chunk.choices:
                    if choice.index == 0:
                        if choice.delta.content:
                            parts.append(choice.delta.content)
                        finish_reason = finish_reason or choice.finish_reason
        except openai.APIError as error:
            _raise_model_error(error, "generation request failed")
        if finish_reason in {"length", "content_filter"}:
            raise ModelOutputError(f"generation ended with finish reason {finish_reason}")
        return GenerateResult(
            text="".join(parts),
            model_reference=ModelReference(
                model_id=self._model_reference.model_id,
                revision=system_fingerprint or self._model_reference.revision,
            ),
        )

    async def close(self) -> None:
        await self._client.close()


class OpenAIEmbedder:
    """Embed text or media through aligned OpenAI-compatible model endpoints."""

    def __init__(
        self,
        query_client: AsyncOpenAI,
        query_model_reference: ModelReference,
        *,
        document_client: AsyncOpenAI,
        document_model_reference: ModelReference,
        space_reference: EmbeddingSpaceReference,
        dimension: int = 1_024,
        request_timeout_seconds: float = 120.0,
        video_frames_per_second: float = DEFAULT_VIDEO_FRAMES_PER_SECOND,
        video_max_pixels: int = DEFAULT_VIDEO_MAX_PIXELS,
    ) -> None:
        if dimension <= 0 or request_timeout_seconds <= 0:
            raise ValueError("dimension and request timeout must be positive")
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
        document_model_id: str,
        document_model_revision: str,
        space_reference: EmbeddingSpaceReference,
        dimension: int = 1_024,
        request_timeout_seconds: float = 120.0,
        max_retries: int = 2,
        video_frames_per_second: float = DEFAULT_VIDEO_FRAMES_PER_SECOND,
        video_max_pixels: int = DEFAULT_VIDEO_MAX_PIXELS,
    ) -> OpenAIEmbedder:
        required = (
            api_key,
            model_id,
            model_revision,
            document_api_key,
            document_model_id,
            document_model_revision,
        )
        if any(not value.strip() for value in required):
            raise ValueError("embedding credentials and model identities are required")
        if not 0 <= max_retries <= 10:
            raise ValueError("max_retries must be between zero and ten")
        return cls(
            AsyncOpenAI(
                api_key=api_key,
                base_url=normalize_base_url(endpoint),
                timeout=request_timeout_seconds,
                max_retries=max_retries,
            ),
            ModelReference(model_id=model_id, revision=model_revision),
            document_client=AsyncOpenAI(
                api_key=document_api_key,
                base_url=normalize_base_url(document_endpoint),
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
            video_frames_per_second=video_frames_per_second,
            video_max_pixels=video_max_pixels,
        )

    @trace_operation("mindbridge.model.embed")
    async def embed(self, request: EmbedRequest) -> EmbedResult:
        """Encode a homogeneous batch without exposing provider request shapes."""
        if not request.inputs:
            return EmbedResult(embeddings=())
        model_reference = (
            self._query_model_reference
            if request.task is EmbedTask.QUERY
            else self._document_model_reference
        )
        set_current_span_attributes(
            {
                "mindbridge.model.id": model_reference.model_id,
                "mindbridge.model.revision": model_reference.revision,
                "mindbridge.embedding.dimension": self._dimension,
                "mindbridge.embedding.input_count": len(request.inputs),
                "mindbridge.embedding.task": request.task.value,
            }
        )
        try:
            if all(_text_only(item) is not None for item in request.inputs):
                vectors = await self._embed_text(request)
            else:
                model_reference = self._query_model_reference
                vectors = tuple(
                    await asyncio.gather(
                        *(self._embed_multimodal(item, request.task) for item in request.inputs)
                    )
                )
        except openai.APIError as error:
            _raise_model_error(error, "embedding request failed")
        return EmbedResult(
            embeddings=tuple(
                Embedding(
                    values=values,
                    model_reference=model_reference,
                    space_reference=self._space_reference,
                )
                for values in vectors
            )
        )

    async def close(self) -> None:
        await self._query_client.close()
        await self._document_client.close()

    async def _embed_text(self, request: EmbedRequest) -> tuple[tuple[float, ...], ...]:
        if request.task is EmbedTask.QUERY:
            client = self._query_client
            model_reference = self._query_model_reference
            prefix = "Query: "
        else:
            client = self._document_client
            model_reference = self._document_model_reference
            prefix = "Document: "
        response = await client.embeddings.create(
            input=[prefix + cast(str, _text_only(item)) for item in request.inputs],
            model=model_reference.model_id,
            dimensions=self._dimension,
            encoding_format="float",
            timeout=self._request_timeout_seconds,
        )
        return _embedding_vectors(
            response,
            model_reference,
            dimension=self._dimension,
            expected_count=len(request.inputs),
        )

    async def _embed_multimodal(
        self,
        input_value: ModelInput,
        task: EmbedTask,
    ) -> tuple[float, ...]:
        response = await self._query_client.post(
            "/embeddings",
            cast_to=CreateEmbeddingResponse,
            body={
                "messages": [
                    {
                        "role": "user",
                        "content": _content_parts(
                            _prefixed_embedding_input(input_value, task),
                            for_embedding=True,
                            video_frames_per_second=self._video_frames_per_second,
                            video_max_pixels=self._video_max_pixels,
                        ),
                    }
                ],
                "model": self._query_model_reference.model_id,
                "dimensions": self._dimension,
                "encoding_format": "float",
            },
            options={"timeout": self._request_timeout_seconds},
        )
        return _embedding_vectors(
            response,
            self._query_model_reference,
            dimension=self._dimension,
            expected_count=1,
        )[0]


def create_generator(config: Mapping[str, object]) -> OpenAIGenerator:
    """Entry-point factory for the bundled OpenAI-compatible generator."""
    _reject_unknown(
        config,
        {
            "api_key",
            "endpoint",
            "model_id",
            "model_revision",
            "request_timeout_seconds",
            "max_retries",
            "reasoning_effort",
            "video_frames_per_second",
            "video_max_pixels",
        },
    )
    return OpenAIGenerator.connect(
        api_key=_string(config, "api_key"),
        endpoint=_string(config, "endpoint"),
        model_id=_string(config, "model_id", DEFAULT_GENERATOR_MODEL_ID),
        model_revision=_string(config, "model_revision"),
        request_timeout_seconds=_float(config, "request_timeout_seconds", 1_800.0),
        max_retries=_integer(config, "max_retries", 2),
        reasoning_effort=cast(
            ReasoningEffort,
            _optional_string(config, "reasoning_effort"),
        ),
        video_frames_per_second=_float(
            config, "video_frames_per_second", DEFAULT_VIDEO_FRAMES_PER_SECOND
        ),
        video_max_pixels=_integer(config, "video_max_pixels", DEFAULT_VIDEO_MAX_PIXELS),
    )


def create_embedder(config: Mapping[str, object]) -> OpenAIEmbedder:
    """Entry-point factory for an aligned OpenAI-compatible embedder pair."""
    _reject_unknown(
        config,
        {
            "api_key",
            "endpoint",
            "model_id",
            "model_revision",
            "document_api_key",
            "document_endpoint",
            "document_model_id",
            "document_model_revision",
            "space_id",
            "space_revision",
            "dimension",
            "request_timeout_seconds",
            "max_retries",
            "video_frames_per_second",
            "video_max_pixels",
        },
    )
    return OpenAIEmbedder.connect(
        api_key=_string(config, "api_key"),
        endpoint=_string(config, "endpoint"),
        model_id=_string(config, "model_id"),
        model_revision=_string(config, "model_revision"),
        document_api_key=_string(config, "document_api_key"),
        document_endpoint=_string(config, "document_endpoint"),
        document_model_id=_string(config, "document_model_id"),
        document_model_revision=_string(config, "document_model_revision"),
        space_reference=EmbeddingSpaceReference(
            space_id=_string(config, "space_id"),
            revision=_string(config, "space_revision"),
        ),
        dimension=_integer(config, "dimension", 1_024),
        request_timeout_seconds=_float(config, "request_timeout_seconds", 120.0),
        max_retries=_integer(config, "max_retries", 2),
        video_frames_per_second=_float(
            config, "video_frames_per_second", DEFAULT_VIDEO_FRAMES_PER_SECOND
        ),
        video_max_pixels=_integer(config, "video_max_pixels", DEFAULT_VIDEO_MAX_PIXELS),
    )


def normalize_base_url(endpoint: str) -> str:
    """Accept an API root or a full compatible chat or embedding endpoint."""
    location = urlsplit(endpoint.strip())
    if (
        location.scheme not in {"http", "https"}
        or not location.netloc
        or location.username is not None
        or location.password is not None
        or location.query
        or location.fragment
    ):
        raise ValueError("endpoint must be an HTTP(S) URL without credentials, query, or fragment")
    path = location.path.rstrip("/")
    for suffix in ("/chat/completions", "/embeddings"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunsplit((location.scheme, location.netloc, path, "", ""))


def _content_parts(
    input_value: ModelInput,
    *,
    for_embedding: bool,
    video_frames_per_second: float,
    video_max_pixels: int,
) -> list[dict[str, object]]:
    content: list[dict[str, object]] = []
    for part in input_value.parts:
        if isinstance(part, TextPart):
            content.append({"type": "text", "text": part.text})
        elif part.kind is MediaKind.IMAGE:
            content.append({"type": "image_url", "image_url": {"url": part.url}})
        elif part.kind is MediaKind.VIDEO:
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": part.url},
                    "fps": part.frames_per_second or video_frames_per_second,
                    "max_pixels": part.max_pixels or video_max_pixels,
                }
            )
        elif for_embedding:
            content.append({"type": "audio_url", "audio_url": {"url": part.url}})
        else:
            source = part.source_uri or part.url
            suffix = PurePosixPath(urlsplit(source).path).suffix
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": part.url,
                        "format": suffix.removeprefix(".").lower() or "wav",
                    },
                }
            )
    return content


def _text_only(input_value: ModelInput) -> str | None:
    return (
        input_value.parts[0].text
        if len(input_value.parts) == 1 and isinstance(input_value.parts[0], TextPart)
        else None
    )


def _prefixed_embedding_input(input_value: ModelInput, task: EmbedTask) -> ModelInput:
    prefix = "Query: " if task is EmbedTask.QUERY else "Document: "
    parts = list(input_value.parts)
    for index, part in enumerate(parts):
        if isinstance(part, TextPart):
            parts[index] = TextPart(prefix + part.text)
            break
    else:
        parts.insert(0, TextPart(prefix.rstrip()))
    return ModelInput(tuple(parts))


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
        _validate_embedding(vector, dimension)
        vectors.append(vector)
    return tuple(vectors)


def _validate_embedding(values: tuple[float, ...], dimension: int) -> None:
    if len(values) != dimension or not all(math.isfinite(value) for value in values):
        raise ModelOutputError("embedding vector has invalid dimension or values")
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-6):
        raise ModelOutputError("embedding vector is not L2-normalized")


def _raise_model_error(error: openai.APIError, message: str) -> None:
    if isinstance(error, openai.APIConnectionError) or (
        isinstance(error, openai.APIStatusError)
        and (error.status_code in {408, 409, 429} or error.status_code >= 500)
    ):
        raise ModelUnavailableError(message) from error
    raise ModelRequestError(message) from error


def _reject_unknown(config: Mapping[str, object], allowed: set[str]) -> None:
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unknown plugin configuration: {', '.join(sorted(unknown))}")


def _string(config: Mapping[str, object], key: str, default: str | None = None) -> str:
    value = config.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be non-empty text")
    return value


def _optional_string(config: Mapping[str, object], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be non-empty text when provided")
    return value


def _integer(config: Mapping[str, object], key: str, default: int) -> int:
    value = config.get(key, default)
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value


def _float(config: Mapping[str, object], key: str, default: float) -> float:
    value = config.get(key, default)
    if type(value) not in {int, float}:
        raise ValueError(f"{key} must be a number")
    return float(cast(int | float, value))
