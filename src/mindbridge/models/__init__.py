"""Model contracts and optional local embedding and speech adapters."""

from mindbridge.models.base import (
    EmbeddingBackend,
    EmbedTask,
    ModelBackend,
    ModelCapabilities,
    ModelInput,
    SpeakerEmbedding,
    SpeechAnalysis,
    SpeechBackend,
    SpeechTurn,
)
from mindbridge.models.funasr import (
    DEFAULT_FUNASR_MODEL_ID,
    DEFAULT_FUNASR_MODEL_REVISION,
    DEFAULT_FUNASR_RECIPE,
    DEFAULT_FUNASR_SPEAKER_MODEL_ID,
    DEFAULT_FUNASR_SPEAKER_REVISION,
    DEFAULT_FUNASR_VAD_MODEL_ID,
    DEFAULT_FUNASR_VAD_REVISION,
    FunASRRecipe,
    FunASRTranscriber,
)
from mindbridge.models.jina import (
    DEFAULT_JINA_DIMENSION,
    DEFAULT_JINA_MODEL_ID,
    DEFAULT_JINA_REVISION,
    JinaOmniEmbedder,
)
from mindbridge.models.sentence_transformers import SentenceTransformersEmbedder

__all__ = [
    "DEFAULT_FUNASR_MODEL_ID",
    "DEFAULT_FUNASR_MODEL_REVISION",
    "DEFAULT_FUNASR_RECIPE",
    "DEFAULT_FUNASR_SPEAKER_MODEL_ID",
    "DEFAULT_FUNASR_SPEAKER_REVISION",
    "DEFAULT_FUNASR_VAD_MODEL_ID",
    "DEFAULT_FUNASR_VAD_REVISION",
    "DEFAULT_JINA_DIMENSION",
    "DEFAULT_JINA_MODEL_ID",
    "DEFAULT_JINA_REVISION",
    "EmbedTask",
    "EmbeddingBackend",
    "FunASRRecipe",
    "FunASRTranscriber",
    "JinaOmniEmbedder",
    "ModelBackend",
    "ModelCapabilities",
    "ModelInput",
    "SentenceTransformersEmbedder",
    "SpeakerEmbedding",
    "SpeechAnalysis",
    "SpeechBackend",
    "SpeechTurn",
]
