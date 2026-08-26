"""OpenAI-compatible adapters for MindBridge's atomic model capabilities."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from html import escape
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import TYPE_CHECKING, Literal, cast
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

import httpx
import openai
from openai import AsyncOpenAI, AsyncStream, Omit, omit
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
from openai.types.completion_usage import CompletionUsage
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.shared import ReasoningEffort
from openai.types.shared_params import ResponseFormatJSONObject, ResponseFormatJSONSchema

from mindbridge.application.capabilities import (
    Embedding,
    EmbedRequest,
    EmbedResult,
    EmbedTask,
    GenerateRequest,
    GenerateResult,
    Generator,
    MediaPart,
    ModelInput,
    TextPart,
)
from mindbridge.configuration import (
    PluginConfigModel,
    PluginInteger,
    PluginNumber,
    PluginText,
)
from mindbridge.core import (
    EmbeddingSpaceReference,
    MediaKind,
    ModelOutputError,
    ModelReference,
    ModelRequestError,
    ModelUnavailableError,
)
from mindbridge.models._vectors import validate_embedding_vector
from mindbridge.models.defaults import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_GENERATOR_MODEL_ID,
    DEFAULT_GENERATOR_REQUEST_TIMEOUT_SECONDS,
    MatryoshkaDimension,
)
from mindbridge.telemetry import log_fields, logger, operation_span, set_current_span_attributes

if TYPE_CHECKING:
    from mindbridge.edge.identity_diarization import FunASRSpeechPipeline, SpeechAnalysis

DEFAULT_VIDEO_FRAMES_PER_SECOND = 1.0
DEFAULT_VIDEO_MAX_PIXELS = 200_704
DEFAULT_ASR_MAXIMUM_MEDIA_BYTES = 64 * 1024 * 1024
_MEDIA_DOWNLOAD_TIMEOUT_SECONDS = 300.0
_MEDIA_SUFFIXES = frozenset(
    {".aac", ".flac", ".m4a", ".mkv", ".mov", ".mp3", ".mp4", ".ogg", ".wav", ".webm"}
)
REASONING_EFFORT_VALUES = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_LOGGER = logger("mindbridge.models.openai")
_SCHEMA_REJECTION_MARKERS = ("response_format", "json_schema", "guided", "structured output")
"""What a provider says when it cannot compile a schema rather than when the request is bad.

A 400 is normally permanent, so an endpoint predating schema support would fail every
observation outright. Matching its complaint is what turns that into one warning and a
fallback; the markers are narrow so an ordinary bad request is still reported as one.
"""


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
        self._schema_decoding_supported = True

    @classmethod
    def connect(
        cls,
        *,
        api_key: str,
        endpoint: str,
        model_id: str = DEFAULT_GENERATOR_MODEL_ID,
        request_timeout_seconds: float = 1_800.0,
        max_retries: int = 2,
        reasoning_effort: ReasoningEffort = None,
        video_frames_per_second: float = DEFAULT_VIDEO_FRAMES_PER_SECOND,
        video_max_pixels: int = DEFAULT_VIDEO_MAX_PIXELS,
    ) -> OpenAIGenerator:
        if any(not value.strip() for value in (api_key, model_id)):
            raise ValueError("API key and model ID are required")
        if not 0 <= max_retries <= 10:
            raise ValueError("max_retries must be between zero and ten")
        return cls(
            AsyncOpenAI(
                api_key=api_key,
                base_url=normalize_base_url(endpoint),
                timeout=request_timeout_seconds,
                max_retries=max_retries,
            ),
            ModelReference(model_id=model_id),
            request_timeout_seconds=request_timeout_seconds,
            reasoning_effort=reasoning_effort,
            video_frames_per_second=video_frames_per_second,
            video_max_pixels=video_max_pixels,
        )

    @operation_span("mindbridge.model.generate")
    async def generate(self, request: GenerateRequest) -> GenerateResult:
        """Stream one deterministic text result and normalize provider failures."""
        constrained = request.output_schema is not None and self._schema_decoding_supported
        set_current_span_attributes(
            {
                "mindbridge.model.id": self._model_reference.model_id,
                "mindbridge.model.input_part_count": len(request.input.parts),
                "mindbridge.model.input.media_count": sum(
                    isinstance(part, MediaPart) for part in request.input.parts
                ),
                "mindbridge.model.json_mode": request.json_mode,
                "mindbridge.model.schema_constrained": constrained,
            }
        )
        messages = cast(
            list[ChatCompletionMessageParam],
            [
                {"role": "system", "content": request.system_prompt},
                {
                    "role": "user",
                    "content": _content_parts(
                        request.input,
                        video_frames_per_second=self._video_frames_per_second,
                        video_max_pixels=self._video_max_pixels,
                    ),
                },
            ],
        )
        request_started_at = perf_counter()
        try:
            completion = await self._complete(request, messages, constrained=constrained)
        except openai.APIError as error:
            _raise_model_error(error, "generation request failed")
        except httpx.HTTPError as error:
            # A provider that drops a long response mid-body raises inside the stream iterator,
            # past the SDK's own error wrapping. Left unclassified it looked permanent, so one
            # transient disconnect failed the whole observation instead of being retried.
            raise ModelUnavailableError("generation request failed") from error
        attributes: dict[str, str | int | float | bool] = {}
        if completion.first_token_at is not None:
            attributes["mindbridge.model.ttft_seconds"] = max(
                0.0, completion.first_token_at - request_started_at
            )
        if completion.system_fingerprint is not None:
            # The serving fingerprint is observability only. Derived records key their stable
            # IDs on model_reference, so a fingerprint that changes when the provider restarts
            # would break idempotent replay of the same consolidation scan.
            attributes["mindbridge.model.system_fingerprint"] = completion.system_fingerprint
        set_current_span_attributes(attributes)
        if completion.finish_reason in {"length", "content_filter"}:
            raise ModelOutputError(
                f"generation ended with finish reason {completion.finish_reason}"
            )
        return GenerateResult(text=completion.text, model_reference=self._model_reference)

    async def close(self) -> None:
        await self._client.close()

    async def _complete(
        self,
        request: GenerateRequest,
        messages: list[ChatCompletionMessageParam],
        *,
        constrained: bool,
    ) -> _StreamedCompletion:
        """Stream one completion, degrading once if this endpoint cannot compile a schema."""
        try:
            return await self._stream(request, messages, constrained=constrained)
        except openai.BadRequestError as error:
            if not constrained or not _rejects_schema_decoding(error):
                raise
            # Latched for the process, not retried per call: an endpoint that cannot compile
            # a schema will not learn to, and paying a rejected request per generation to
            # rediscover that is the cost this flag exists to avoid.
            self._schema_decoding_supported = False
            set_current_span_attributes({"mindbridge.model.schema_constrained": False})
            _LOGGER.warning(
                "endpoint rejected schema-constrained decoding, falling back to JSON mode",
                extra=log_fields(
                    model_id=self._model_reference.model_id,
                    schema=request.output_schema.name if request.output_schema else None,
                    status_code=error.status_code,
                ),
            )
        return await self._stream(request, messages, constrained=False)

    async def _stream(
        self,
        request: GenerateRequest,
        messages: list[ChatCompletionMessageParam],
        *,
        constrained: bool,
    ) -> _StreamedCompletion:
        stream = await self._client.chat.completions.create(
            model=self._model_reference.model_id,
            messages=messages,
            modalities=["text"],
            max_tokens=request.max_output_tokens,
            response_format=_response_format(request, constrained=constrained),
            reasoning_effort=(
                self._reasoning_effort if self._reasoning_effort is not None else omit
            ),
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
            timeout=self._request_timeout_seconds,
        )
        return await _consume_completion_stream(stream)


class AudioFallbackGenerator:
    """Diarize AV for a VLM while leaving native Omni generation untouched."""

    def __init__(
        self,
        generator: OpenAIGenerator,
        *,
        asr_model_id: str | None = None,
        asr_device: str | None = None,
        maximum_media_bytes: int = DEFAULT_ASR_MAXIMUM_MEDIA_BYTES,
    ) -> None:
        if maximum_media_bytes <= 0:
            raise ValueError("ASR media limit must be positive")
        self._generator = generator
        self._asr_model_id = asr_model_id
        self._asr_device = asr_device
        self._maximum_media_bytes = maximum_media_bytes
        self._pipeline: FunASRSpeechPipeline | None = None
        # ponytail: one local model lock; use one worker per device if ASR throughput matters.
        self._pipeline_lock = asyncio.Lock()

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        if not any(
            isinstance(part, MediaPart) and part.kind in {MediaKind.AUDIO, MediaKind.VIDEO}
            for part in request.input.parts
        ):
            return await self._generator.generate(request)

        parts: list[TextPart | MediaPart] = []
        for part in request.input.parts:
            if not isinstance(part, MediaPart) or part.kind not in {
                MediaKind.AUDIO,
                MediaKind.VIDEO,
            }:
                parts.append(part)
                continue
            transcript = await self._transcribe(part)
            if part.kind is MediaKind.VIDEO:
                parts.append(part)
            parts.append(_asr_transcript_part(part, transcript))
        return await self._generator.generate(replace(request, input=ModelInput(tuple(parts))))

    async def close(self) -> None:
        await self._generator.close()

    async def _transcribe(self, part: MediaPart) -> str:
        suffix = _media_suffix(part)
        with TemporaryDirectory(prefix="mindbridge-asr-") as directory:
            path = Path(directory, f"input{suffix}")
            data = await asyncio.to_thread(_read_media, part.url, self._maximum_media_bytes)
            await asyncio.to_thread(path.write_bytes, data)
            async with self._pipeline_lock:
                if self._pipeline is None:
                    self._pipeline = await asyncio.to_thread(
                        _load_funasr,
                        self._asr_model_id,
                        self._asr_device,
                    )
                try:
                    return _speaker_transcript(await self._pipeline.analyze_file(path))
                except ModelOutputError:
                    raise
                except Exception as error:
                    raise ModelUnavailableError("FunASR transcription failed") from error


def _load_funasr(model_id: str | None, device: str | None) -> FunASRSpeechPipeline:
    try:
        from mindbridge.edge.identity_diarization import FunASRSpeechPipeline
    except ImportError as error:
        raise ModelUnavailableError(
            "install MindBridge with the edge extra to use generator audio_mode='transcribe'"
        ) from error
    try:
        return FunASRSpeechPipeline.load(device=device, model_id=model_id)
    except ModelUnavailableError:
        raise
    except Exception as error:
        raise ModelUnavailableError("FunASR could not be loaded") from error


def _speaker_transcript(analysis: SpeechAnalysis) -> str:
    speakers: dict[str | None, str] = {}
    lines = []
    for segment in analysis.segments:
        speaker = speakers.setdefault(segment.speaker_label, _speaker_name(len(speakers)))
        lines.append(
            f"speaker {speaker} | {_timestamp(segment.start_ms)}-{_timestamp(segment.end_ms)} | "
            f"{segment.transcript}"
        )
    return "\n".join(lines)


def _speaker_name(index: int) -> str:
    return chr(ord("A") + index) if index < 26 else f"S{index + 1}"


def _timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"


def _asr_transcript_part(part: MediaPart, transcript: str) -> TextPart:
    source = escape(part.source_uri or part.kind.value, quote=True)
    content = escape(transcript or "[no speech detected]")
    return TextPart(
        f'<asr_transcript source_uri="{source}" trust="untrusted">{content}</asr_transcript>'
    )


def _media_suffix(part: MediaPart) -> str:
    suffix = PurePosixPath(urlsplit(part.source_uri or part.url).path).suffix.lower()
    if suffix in _MEDIA_SUFFIXES:
        return suffix
    return ".mp4" if part.kind is MediaKind.VIDEO else ".wav"


def _read_media(url: str, maximum_bytes: int) -> bytes:
    location = urlsplit(url)
    if not location.scheme:
        try:
            return _read_bounded_file(Path(url), maximum_bytes)
        except (OSError, ValueError) as error:
            raise ModelRequestError("local ASR media could not be read") from error
    if (
        location.scheme not in {"data", "file", "http", "https"}
        or (location.scheme in {"http", "https"} and not location.netloc)
        or (location.scheme == "file" and location.netloc not in {"", "localhost"})
        or location.username is not None
        or location.password is not None
        or location.fragment
        or (location.scheme == "file" and location.query)
        or (location.scheme == "data" and len(url) > 4 * ((maximum_bytes + 2) // 3) + 256)
    ):
        raise ModelRequestError("ASR media must use data, file, HTTP, or HTTPS")
    try:
        with urlopen(url, timeout=_MEDIA_DOWNLOAD_TIMEOUT_SECONDS) as response:
            declared = response.headers.get("content-length")
            if declared is not None and int(declared) > maximum_bytes:
                raise ModelRequestError(f"ASR media exceeds {maximum_bytes} bytes")
            data = cast(bytes, response.read(maximum_bytes + 1))
    except ModelRequestError:
        raise
    except (OSError, ValueError) as error:
        failure = (
            ModelUnavailableError if location.scheme in {"http", "https"} else ModelRequestError
        )
        raise failure("ASR media could not be read") from error
    if len(data) > maximum_bytes:
        raise ModelRequestError(f"ASR media exceeds {maximum_bytes} bytes")
    return data


def _read_bounded_file(path: Path, maximum_bytes: int) -> bytes:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size > maximum_bytes:
        raise ValueError("media is not a bounded regular file")
    return resolved.read_bytes()


class OpenAIEmbedder:
    """Embed text or media through one OpenAI-compatible Omni endpoint."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model_reference: ModelReference,
        *,
        space_reference: EmbeddingSpaceReference,
        dimension: int = 1_024,
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
        model_id: str,
        space_reference: EmbeddingSpaceReference,
        dimension: int = 1_024,
        request_timeout_seconds: float = 120.0,
        max_retries: int = 2,
    ) -> OpenAIEmbedder:
        if any(not value.strip() for value in (api_key, model_id)):
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
            ModelReference(model_id=model_id),
            space_reference=space_reference,
            dimension=dimension,
            request_timeout_seconds=request_timeout_seconds,
        )

    @property
    def space_reference(self) -> EmbeddingSpaceReference:
        """Declare the search space both aligned endpoints write into."""
        return self._space_reference

    @operation_span("mindbridge.model.embed")
    async def embed(self, request: EmbedRequest) -> EmbedResult:
        """Encode a homogeneous batch without exposing provider request shapes."""
        if not request.inputs:
            return EmbedResult(embeddings=())
        set_current_span_attributes(
            {
                "mindbridge.model.id": self._model_reference.model_id,
                "mindbridge.embedding.space_id": self._space_reference.space_id,
                "mindbridge.embedding.dimension": self._dimension,
                "mindbridge.embedding.input_count": len(request.inputs),
                "mindbridge.embedding.task": request.task.value,
            }
        )
        try:
            if all(_text_only(item) is not None for item in request.inputs):
                vectors, charged_tokens = await self._embed_text(request)
            else:
                vectors, charged_tokens = await self._embed_multimodal(request)
        except openai.APIError as error:
            _raise_model_error(error, "embedding request failed")
        # Serving the encoder turns embedding into a metered call, so its cost has to reach the
        # same account generation already reports into. An embedding charges entirely on its
        # input -- there are no completion tokens to report -- so what lands here is the whole
        # batch's bill.
        set_current_span_attributes(
            {
                "mindbridge.model.input_tokens": charged_tokens,
                "mindbridge.model.total_tokens": charged_tokens,
            }
        )
        return EmbedResult(
            embeddings=tuple(
                Embedding(
                    values=values,
                    model_reference=self._model_reference,
                    space_reference=self._space_reference,
                )
                for values in vectors
            )
        )

    async def close(self) -> None:
        await self._client.close()

    async def _embed_text(
        self,
        request: EmbedRequest,
    ) -> tuple[tuple[tuple[float, ...], ...], int]:
        prefix = "Query: " if request.task is EmbedTask.QUERY else "Document: "
        response = await self._client.embeddings.create(
            input=[prefix + cast(str, _text_only(item)) for item in request.inputs],
            model=self._model_reference.model_id,
            dimensions=self._dimension,
            encoding_format="float",
            timeout=self._request_timeout_seconds,
        )
        return (
            _embedding_vectors(
                response,
                self._model_reference,
                dimension=self._dimension,
                expected_count=len(request.inputs),
            ),
            response.usage.prompt_tokens,
        )

    async def _embed_multimodal(
        self,
        request: EmbedRequest,
    ) -> tuple[tuple[tuple[float, ...], ...], int]:
        response = await self._client.post(
            "/embeddings",
            cast_to=CreateEmbeddingResponse,
            body={
                "input": [
                    [
                        {
                            "role": "user",
                            "content": _embedding_content_parts(
                                _prefixed_embedding_input(input_value, request.task)
                            ),
                        }
                    ]
                    for input_value in request.inputs
                ],
                "model": self._model_reference.model_id,
                "dimensions": self._dimension,
                "encoding_format": "float",
            },
            options={"timeout": self._request_timeout_seconds},
        )
        return (
            _embedding_vectors(
                response,
                self._model_reference,
                dimension=self._dimension,
                expected_count=len(request.inputs),
            ),
            response.usage.prompt_tokens,
        )


class _GeneratorConfig(PluginConfigModel):
    api_key: PluginText
    endpoint: PluginText
    model_id: PluginText = DEFAULT_GENERATOR_MODEL_ID
    request_timeout_seconds: PluginNumber = DEFAULT_GENERATOR_REQUEST_TIMEOUT_SECONDS
    max_retries: PluginInteger = 2
    reasoning_effort: PluginText | None = None
    video_frames_per_second: PluginNumber = DEFAULT_VIDEO_FRAMES_PER_SECOND
    video_max_pixels: PluginInteger = DEFAULT_VIDEO_MAX_PIXELS
    audio_mode: Literal["native", "transcribe"] = "native"
    asr_model_id: PluginText | None = None
    asr_device: PluginText | None = None


class _EmbedderConfig(PluginConfigModel):
    api_key: PluginText
    endpoint: PluginText
    model_id: PluginText
    space_id: PluginText
    dimension: MatryoshkaDimension = DEFAULT_EMBEDDING_DIMENSION
    request_timeout_seconds: PluginNumber = 120.0
    max_retries: PluginInteger = 2


def create_generator(config: Mapping[str, object]) -> Generator:
    """Entry-point factory for the bundled OpenAI-compatible generator."""
    validated = _GeneratorConfig.model_validate(config)
    generator = OpenAIGenerator.connect(
        api_key=validated.api_key,
        endpoint=validated.endpoint,
        model_id=validated.model_id,
        request_timeout_seconds=validated.request_timeout_seconds,
        max_retries=validated.max_retries,
        reasoning_effort=cast(ReasoningEffort, validated.reasoning_effort),
        video_frames_per_second=validated.video_frames_per_second,
        video_max_pixels=validated.video_max_pixels,
    )
    if validated.audio_mode == "native":
        return generator
    return AudioFallbackGenerator(
        generator,
        asr_model_id=validated.asr_model_id,
        asr_device=validated.asr_device,
    )


def create_embedder(config: Mapping[str, object]) -> OpenAIEmbedder:
    """Entry-point factory for the OpenAI-compatible Omni embedder."""
    validated = _EmbedderConfig.model_validate(config)
    return OpenAIEmbedder.connect(
        api_key=validated.api_key,
        endpoint=validated.endpoint,
        model_id=validated.model_id,
        space_reference=EmbeddingSpaceReference(space_id=validated.space_id),
        dimension=validated.dimension,
        request_timeout_seconds=validated.request_timeout_seconds,
        max_retries=validated.max_retries,
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


@dataclass(frozen=True, slots=True)
class _StreamedCompletion:
    """The first choice of one streamed completion and its serving metadata."""

    text: str
    finish_reason: str | None
    system_fingerprint: str | None
    first_token_at: float | None


async def _consume_completion_stream(
    stream: AsyncStream[ChatCompletionChunk],
) -> _StreamedCompletion:
    parts: list[str] = []
    finish_reason: str | None = None
    system_fingerprint: str | None = None
    first_token_at: float | None = None
    charged: CompletionUsage | None = None
    try:
        async for chunk in stream:
            system_fingerprint = system_fingerprint or chunk.system_fingerprint
            # Kept rather than reported here: the token account accumulates what it is given,
            # and a server with `continuous_usage_stats` repeats the running total on every
            # chunk, which would bill the request once per chunk. Each report supersedes the
            # last, so the final one is the charge.
            charged = chunk.usage or charged
            for choice in chunk.choices:
                if choice.index != 0:
                    continue
                if choice.delta.content:
                    if first_token_at is None:
                        first_token_at = perf_counter()
                    parts.append(choice.delta.content)
                finish_reason = finish_reason or choice.finish_reason
    finally:
        # Reported even when the stream drops mid-body: the partial response was still billed,
        # and the failed attempt carries that charge into its job row.
        if charged is not None:
            set_current_span_attributes(
                {
                    "mindbridge.model.input_tokens": charged.prompt_tokens,
                    "mindbridge.model.output_tokens": charged.completion_tokens,
                    "mindbridge.model.total_tokens": charged.total_tokens,
                }
            )
    return _StreamedCompletion(
        text="".join(parts),
        finish_reason=finish_reason,
        system_fingerprint=system_fingerprint,
        first_token_at=first_token_at,
    )


def _content_parts(
    input_value: ModelInput,
    *,
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


def _embedding_content_parts(input_value: ModelInput) -> list[dict[str, object]]:
    """Render the bundled SentenceTransformers service contract."""
    content: list[dict[str, object]] = []
    for part in input_value.parts:
        if isinstance(part, TextPart):
            content.append({"type": "text", "text": part.text})
        else:
            kind = {
                MediaKind.IMAGE: "image_url",
                MediaKind.VIDEO: "video_url",
                MediaKind.AUDIO: "audio_url",
            }[part.kind]
            content.append({"type": kind, kind: {"url": part.url}})
    return content


def _response_format(
    request: GenerateRequest,
    *,
    constrained: bool,
) -> ResponseFormatJSONSchema | ResponseFormatJSONObject | Omit:
    """Ask for the strongest output contract this request and endpoint both support."""
    schema = request.output_schema
    if constrained and schema is not None:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.name,
                "schema": cast(dict[str, object], json.loads(schema.json_schema)),
                "strict": True,
            },
        }
    return {"type": "json_object"} if request.json_mode or schema is not None else omit


def _rejects_schema_decoding(error: openai.BadRequestError) -> bool:
    """Tell a provider without schema support apart from a request that is simply invalid."""
    message = str(error).casefold()
    return any(marker in message for marker in _SCHEMA_REJECTION_MARKERS)


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
        validate_embedding_vector(vector, dimension)
        vectors.append(vector)
    return tuple(vectors)


def _raise_model_error(error: openai.APIError, message: str) -> None:
    transient = isinstance(error, openai.APIConnectionError) or (
        isinstance(error, openai.APIStatusError)
        and (error.status_code in {408, 409, 429} or error.status_code >= 500)
    )
    # This classification decides whether a long run retries or dies, and both outcomes
    # otherwise reach the operator as the same one-line message with the status code gone.
    _LOGGER.warning(
        "provider request failed",
        extra=log_fields(
            error_type=type(error).__name__,
            status_code=getattr(error, "status_code", None),
            retryable=transient,
        ),
    )
    if transient:
        raise ModelUnavailableError(message) from error
    raise ModelRequestError(message) from error
