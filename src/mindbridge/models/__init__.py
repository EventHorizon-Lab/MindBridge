"""Model contracts and optional local embedding and speech adapters."""

from mindbridge.models.base import (
    EmbeddingBackend,
    EmbedTask,
    FaceAnalysis,
    FaceBackend,
    FaceDetection,
    FaceEmbedding,
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
from mindbridge.models.insightface import (
    DEFAULT_INSIGHTFACE_MODEL,
    DEFAULT_INSIGHTFACE_MODEL_REVISION,
    InsightFaceRecognizer,
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
    "DEFAULT_INSIGHTFACE_MODEL",
    "DEFAULT_INSIGHTFACE_MODEL_REVISION",
    "DEFAULT_JINA_DIMENSION",
    "DEFAULT_JINA_MODEL_ID",
    "DEFAULT_JINA_REVISION",
    "EmbedTask",
    "EmbeddingBackend",
    "FaceAnalysis",
    "FaceBackend",
    "FaceDetection",
    "FaceEmbedding",
    "FunASRRecipe",
    "FunASRTranscriber",
    "InsightFaceRecognizer",
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
