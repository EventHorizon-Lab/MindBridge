"""Typed declarative composition over MindBridge's existing backend protocols."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
)
from pydantic import (
    ValidationError as PydanticValidationError,
)

import mindbridge.recipes as recipes
from mindbridge.exceptions import ValidationError
from mindbridge.models.base import (
    EmbeddingBackend,
    FaceBackend,
    FormationBackend,
    GenerationBackend,
    SpeechBackend,
    TranscriptionBackend,
)
from mindbridge.models.funasr import FunASRTranscriber
from mindbridge.models.jina import DEFAULT_JINA_DIMENSION, JinaOmniEmbedder
from mindbridge.models.openai_sdk import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_TRANSCRIPTION_MODEL,
    OpenAIModels,
)
from mindbridge.models.opencv_face import OpenCVFaceAnalyzer
from mindbridge.models.sentence_transformers import SentenceTransformersEmbedder
from mindbridge.plugins import MemoryConfig, MemoryPlugins, MemorySettings
from mindbridge.types import Modality

_Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_PositiveFloat = Annotated[float, Field(strict=True, gt=0)]
_PositiveInt = Annotated[int, Field(strict=True, gt=0)]
_NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
_Temperature = Annotated[float, Field(strict=True, ge=0, le=2)]
_Seed = Annotated[int, Field(strict=True, ge=0, lt=2**63)]
_UnitInterval = Annotated[float, Field(strict=True, ge=0, le=1)]
_DISCRIMINATED_SLOTS = frozenset({"embedding", "speech"})


class _ConfigModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


class _OpenAIConfig(_ConfigModel):
    provider: Literal["openai"]
    base_url: _Text | None = None
    timeout: _PositiveFloat | None = None
    max_retries: _NonNegativeInt | None = None


class OpenAIEmbeddingConfig(_OpenAIConfig):
    model: _Text = DEFAULT_EMBEDDING_MODEL
    dimension: _PositiveInt = DEFAULT_EMBEDDING_DIMENSION
    space: _Text | None = None
    # An OpenAI-compatible server may host a multimodal embedding model. Declaring which
    # modalities it accepts is what lets routing send image, video, and audio memories to it
    # instead of failing the write; `messages` is the request shape those servers read media
    # parts from. Both already existed on the backend and were unreachable from configuration.
    # At least one: an empty set declares a model that can embed nothing, which builds a `Memory`
    # whose every write fails with "does not support: text". Configuration rejects out-of-range
    # values rather than deferring them to the first `add`.
    modalities: Annotated[frozenset[Modality], Field(min_length=1)] = frozenset({Modality.TEXT})
    request_format: Literal["input", "messages"] = "input"


class JinaEmbeddingConfig(_ConfigModel):
    provider: Literal["jina-omni"]
    dimension: _PositiveInt = DEFAULT_JINA_DIMENSION
    device: _Text | None = None
    batch_size: _PositiveInt = 32


class SentenceTransformersEmbeddingConfig(_ConfigModel):
    provider: Literal["sentence-transformers"]
    model: _Text
    revision: _Text
    dimension: _PositiveInt | None = None
    device: _Text | None = None
    batch_size: _PositiveInt = 32


class OpenAIGenerationConfig(_OpenAIConfig):
    model: _Text = DEFAULT_GENERATION_MODEL
    modalities: frozenset[Modality] = frozenset({Modality.TEXT})
    temperature: _Temperature | None = None
    seed: _Seed | None = None
    max_tokens: _PositiveInt | None = None
    video_limit: _PositiveInt | None = 8
    extra_body: Mapping[str, object] | None = None


class OpenAIFormationConfig(_OpenAIConfig):
    """Chat-completion knobs for the adapter that proposes derived typed memories.

    The bundled adapter derives its formation contract from the same completion controls it uses
    to answer, so this slot repeats them rather than reusing the generation slot: formation is a
    separate LLM round-trip on the write path and usually wants its own model, token budget, and
    endpoint. Leaving the slot out keeps that round-trip off, which is the default.
    """

    model: _Text = DEFAULT_GENERATION_MODEL
    # `formation_capabilities` gates which observations reach the former at all: an observation
    # whose modalities are not covered here is skipped, silently and by design. Declare the media
    # the endpoint really accepts or image and video sources will never form anything.
    modalities: frozenset[Modality] = frozenset({Modality.TEXT})
    temperature: _Temperature | None = None
    seed: _Seed | None = None
    # Formation returns JSON; a truncated response is a hard error rather than a partial parse.
    max_tokens: _PositiveInt | None = None
    extra_body: Mapping[str, object] | None = None


class OpenAITranscriptionConfig(_OpenAIConfig):
    model: _Text = DEFAULT_TRANSCRIPTION_MODEL
    space: _Text | None = None


class FunASRSpeechConfig(_ConfigModel):
    provider: Literal["funasr"]
    device: _Text = "auto"


class OpenCVFaceConfig(_ConfigModel):
    provider: Literal["opencv"]
    detector_model: Path
    recognizer_model: Path
    score_threshold: _UnitInterval = 0.9
    nms_threshold: _UnitInterval = 0.3
    top_k: _PositiveInt = 5000
    frame_interval_ms: _PositiveInt = 1000
    max_video_frames: _PositiveInt = 300


EmbeddingProviderConfig = Annotated[
    OpenAIEmbeddingConfig | JinaEmbeddingConfig | SentenceTransformersEmbeddingConfig,
    Field(discriminator="provider"),
]
SpeechProviderConfig = Annotated[
    OpenAITranscriptionConfig | FunASRSpeechConfig,
    Field(discriminator="provider"),
]


class MindBridgeConfig(_ConfigModel):
    """Pure-data configuration for MindBridge's bundled backend adapters."""

    data_dir: Path = Path(".mindbridge")
    embedding: EmbeddingProviderConfig
    generation: OpenAIGenerationConfig | None = None
    # Omitted by default: formation adds an LLM round-trip to every write, and derived memories
    # are a union with the raw sources rather than a replacement for them.
    formation: OpenAIFormationConfig | None = None
    speech: SpeechProviderConfig | None = None
    face: OpenCVFaceConfig | None = None
    settings: MemorySettings = Field(default_factory=MemorySettings)


@dataclass(frozen=True, slots=True)
class MemoryComposition:
    """Constructed plugins and settings owned by one declarative composition."""

    data_dir: Path
    plugins: MemoryPlugins
    settings: MemoryConfig

    def close(self) -> None:
        seen: set[int] = set()
        resources = (
            self.plugins.face_analyzer,
            self.plugins.vision_describer,
            self.plugins.transcriber,
            self.plugins.former,
            self.plugins.answerer,
            self.plugins.embedder,
        )
        for resource in resources:
            if resource is None or id(resource) in seen:
                continue
            seen.add(id(resource))
            with suppress(Exception):
                resource.close()


def resolve_memory_config(
    value: MindBridgeConfig | Mapping[str, object],
) -> MemoryComposition:
    """Construct and return the bundled plugins described by declarative configuration."""
    config = _validated_config(value)
    with ExitStack() as cleanup:
        embedder = _build_embedding(config.embedding)
        cleanup.callback(embedder.close)
        answerer = None if config.generation is None else _build_generation(config.generation)
        if answerer is not None:
            cleanup.callback(answerer.close)
        former = None if config.formation is None else _build_formation(config.formation)
        if former is not None:
            cleanup.callback(former.close)
        transcriber = None if config.speech is None else _build_speech(config.speech)
        if transcriber is not None:
            cleanup.callback(transcriber.close)
        face = None if config.face is None else _build_face(config.face)
        if face is not None:
            cleanup.callback(face.close)
        plugins = MemoryPlugins(
            embedder=embedder,
            answerer=answerer,
            transcriber=transcriber,
            face_analyzer=face,
            former=former,
        )
        cleanup.pop_all()
    return MemoryComposition(config.data_dir, plugins, config.settings)


def _validated_config(value: MindBridgeConfig | Mapping[str, object]) -> MindBridgeConfig:
    if isinstance(value, MindBridgeConfig):
        return value
    if not isinstance(value, Mapping):
        raise ValidationError("config must be a MindBridgeConfig or mapping")
    try:
        return MindBridgeConfig.model_validate(value)
    except PydanticValidationError as error:
        issues = []
        for issue in error.errors(include_url=False, include_context=False, include_input=False):
            location = tuple(issue["loc"])
            if issue["type"] in {"union_tag_invalid", "union_tag_not_found"}:
                location = (*location, "provider")
            elif len(location) >= 2 and location[0] in _DISCRIMINATED_SLOTS:
                location = (location[0], *location[2:])
            path = ".".join(("config", *(str(part) for part in location)))
            issues.append(f"{path}: {issue['msg']}")
        raise ValidationError("; ".join(issues)) from None


def _openai_factory(values: dict[str, object]) -> OpenAIModels:
    factory = cast(Callable[..., OpenAIModels], recipes._owned_openai_models)
    return factory(**values)


def _openai_values(config: _OpenAIConfig) -> dict[str, object]:
    return {
        key: value
        for key, value in {
            "base_url": config.base_url,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
        }.items()
        if value is not None
    }


def _build_embedding(config: EmbeddingProviderConfig) -> EmbeddingBackend:
    if isinstance(config, OpenAIEmbeddingConfig):
        values = _openai_values(config)
        values.update(
            embedding_model=config.model,
            embedding_dimension=config.dimension,
            embedding_capabilities=config.modalities,
            embedding_request_format=config.request_format,
        )
        if config.space is not None:
            values["embedding_space"] = config.space
        return cast(EmbeddingBackend, _openai_factory(values))
    if isinstance(config, JinaEmbeddingConfig):
        return JinaOmniEmbedder(
            dimension=config.dimension,
            device=config.device,
            batch_size=config.batch_size,
        )
    return SentenceTransformersEmbedder.load(
        config.model,
        revision=config.revision,
        dimension=config.dimension,
        device=config.device,
        batch_size=config.batch_size,
    )


def _completion_values(
    config: OpenAIGenerationConfig | OpenAIFormationConfig,
) -> dict[str, object]:
    values = _openai_values(config)
    values.update(
        generation_model=config.model,
        generation_capabilities=config.modalities,
    )
    optional = {
        "generation_temperature": config.temperature,
        "generation_seed": config.seed,
        "generation_max_tokens": config.max_tokens,
        "generation_extra_body": config.extra_body,
    }
    values.update((key, value) for key, value in optional.items() if value is not None)
    return values


def _build_generation(config: OpenAIGenerationConfig) -> GenerationBackend:
    values = _completion_values(config)
    if config.video_limit != 8:
        values["generation_video_limit"] = config.video_limit
    return cast(GenerationBackend, _openai_factory(values))


def _build_formation(config: OpenAIFormationConfig) -> FormationBackend:
    # The bundled adapter reads the generation controls for `form`; `video_limit` is answer-only.
    return cast(FormationBackend, _openai_factory(_completion_values(config)))


def _build_speech(config: SpeechProviderConfig) -> SpeechBackend | TranscriptionBackend:
    if isinstance(config, FunASRSpeechConfig):
        return FunASRTranscriber(device=config.device)
    values = _openai_values(config)
    values["transcription_model"] = config.model
    if config.space is not None:
        values["transcription_space"] = config.space
    return cast(TranscriptionBackend, _openai_factory(values))


def _build_face(config: OpenCVFaceConfig) -> FaceBackend:
    return OpenCVFaceAnalyzer(
        config.detector_model,
        config.recognizer_model,
        score_threshold=config.score_threshold,
        nms_threshold=config.nms_threshold,
        top_k=config.top_k,
        frame_interval_ms=config.frame_interval_ms,
        max_video_frames=config.max_video_frames,
    )


__all__ = [
    "EmbeddingProviderConfig",
    "FunASRSpeechConfig",
    "JinaEmbeddingConfig",
    "MemoryComposition",
    "MindBridgeConfig",
    "OpenAIEmbeddingConfig",
    "OpenAIFormationConfig",
    "OpenAIGenerationConfig",
    "OpenAITranscriptionConfig",
    "OpenCVFaceConfig",
    "SentenceTransformersEmbeddingConfig",
    "SpeechProviderConfig",
    "resolve_memory_config",
]
