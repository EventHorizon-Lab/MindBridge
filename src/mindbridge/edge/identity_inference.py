"""Thin InsightFace and device-local identity inference adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

from mindbridge.contracts import IdentityObservationInput
from mindbridge.core import (
    IdentityKind,
    IdentityScope,
    ModelOutputError,
    ModelReference,
    ModelUnavailableError,
    derive_stable_id,
)
from mindbridge.edge.identity import LocalIdentitySample, SQLiteIdentityMemory
from mindbridge.models.compute import select_onnx_providers

INSIGHTFACE_MODEL = ModelReference(model_id="insightface/buffalo_l", revision="1.0.1")
ERES2NETV2_MODEL = ModelReference(
    model_id="iic/speech_eres2netv2_sv_zh-cn_16k-common",
    revision="v1.0.1",
)
OBSERVATION_SCOPED_VOICE_MODEL = ModelReference(
    model_id="mindbridge/observation-scoped-voice",
    revision="1",
)


class _FaceAnalysis(Protocol):
    def prepare(self, *, ctx_id: int, det_size: tuple[int, int]) -> None: ...

    def get(self, image: object) -> list[object]: ...


@dataclass(frozen=True, slots=True)
class FaceEmbeddingSample:
    """One timestamped face anchor produced from the native video stream."""

    sample_id: str
    start_ms: int
    end_ms: int
    confidence: float
    visual_bbox_xyxy: tuple[float, float, float, float]
    embedding: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.sample_id.strip() or self.start_ms < 0 or self.end_ms < self.start_ms:
            raise ValueError("face sample must have an ID and ordered time range")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("face sample confidence must be between 0 and 1")
        left, top, right, bottom = self.visual_bbox_xyxy
        if (
            not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in self.visual_bbox_xyxy)
            or right <= left
            or bottom <= top
        ):
            raise ValueError("face sample bounding box must be normalized xyxy")
        _normalized_embedding(self.embedding)


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """One diarizer-produced speaker turn to encode locally."""

    sample_id: str
    start_ms: int
    end_ms: int
    confidence: float = 1.0
    transcript: str | None = None
    speaker_label: str | None = None

    def __post_init__(self) -> None:
        if not self.sample_id.strip() or self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("speech segment must have an ID and positive time range")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("speech segment confidence must be between 0 and 1")
        if self.transcript is not None and (
            not self.transcript.strip() or self.transcript != self.transcript.strip()
        ):
            raise ValueError("speech segment transcript must not be blank or padded")
        if self.speaker_label is not None and not self.speaker_label.strip():
            raise ValueError("speech segment speaker_label must not be blank")


@dataclass(frozen=True, slots=True)
class SpeakerEmbeddingSample:
    """One upstream diarization centroid retained only in device memory."""

    speaker_label: str
    embedding: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.speaker_label.strip():
            raise ValueError("speaker label must not be blank")
        _normalized_embedding(self.embedding)


class InsightFaceVideoEncoder:
    """Run the upstream RetinaFace/ArcFace pack on sampled video frames."""

    def __init__(
        self,
        face_analysis: _FaceAnalysis,
        *,
        execution_providers: tuple[str, ...],
        minimum_detection_confidence: float = 0.6,
        minimum_embedding_norm: float = 20.0,
        minimum_face_pixels: int = 32,
    ) -> None:
        if (
            not 0.0 <= minimum_detection_confidence <= 1.0
            or minimum_embedding_norm <= 0
            or minimum_face_pixels <= 0
        ):
            raise ValueError("face detection thresholds are invalid")
        if not execution_providers:
            raise ValueError("face inference requires an execution provider")
        self._face_analysis = face_analysis
        self.execution_providers = execution_providers
        self._minimum_detection_confidence = minimum_detection_confidence
        self._minimum_embedding_norm = minimum_embedding_norm
        self._minimum_face_pixels = minimum_face_pixels

    @classmethod
    def load(
        cls,
        *,
        device: str | None = None,
        detection_size: tuple[int, int] = (640, 640),
        minimum_detection_confidence: float = 0.6,
        minimum_embedding_norm: float = 20.0,
        minimum_face_pixels: int = 32,
    ) -> InsightFaceVideoEncoder:
        """Load InsightFace on CUDA by default, or TensorRT when explicitly requested."""
        try:
            insightface = import_module("insightface.app")
            onnxruntime = import_module("onnxruntime")
        except ImportError as error:
            raise ModelUnavailableError(
                "install InsightFace, ONNX Runtime, and OpenCV for face inference"
            ) from error
        providers = select_onnx_providers(
            tuple(cast(list[str], onnxruntime.get_available_providers())),
            device,
        )
        preload_dlls = getattr(onnxruntime, "preload_dlls", None)
        if "CUDAExecutionProvider" in providers and callable(preload_dlls):
            preload_dlls()
        face_analysis = cast(
            _FaceAnalysis,
            insightface.FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection", "recognition"],
                providers=list(providers),
            ),
        )
        face_analysis.prepare(
            ctx_id=0 if providers[0] != "CPUExecutionProvider" else -1,
            det_size=detection_size,
        )
        applied_providers = _applied_face_providers(face_analysis)
        requested = (device or "auto").strip().lower()
        required_provider = {
            "cuda": "CUDAExecutionProvider",
            "tensorrt": "TensorrtExecutionProvider",
        }.get(requested)
        if required_provider is not None and required_provider not in applied_providers:
            raise ModelUnavailableError(f"requested ONNX {requested} provider failed to load")
        return cls(
            face_analysis,
            execution_providers=applied_providers,
            minimum_detection_confidence=minimum_detection_confidence,
            minimum_embedding_norm=minimum_embedding_norm,
            minimum_face_pixels=minimum_face_pixels,
        )

    def encode_frame(
        self,
        image: object,
        *,
        timestamp_ms: int,
        duration_ms: int,
    ) -> tuple[FaceEmbeddingSample, ...]:
        """Consume one native BGR frame from an existing OpenCV/GStreamer appsink."""
        if timestamp_ms < 0 or duration_ms <= 0:
            raise ValueError("frame timestamp and duration must be positive and ordered")
        try:
            height, width = cast(Any, image).shape[:2]
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("face inference requires an image with height and width") from error
        if height <= 0 or width <= 0:
            raise ValueError("face inference requires a non-empty image")
        samples = []
        for face_index, face in enumerate(self._face_analysis.get(image)):
            confidence = float(cast(Any, face).det_score)
            left, top, right, bottom = (float(value) for value in cast(Any, face).bbox.tolist())
            if (
                confidence < self._minimum_detection_confidence
                or right - left < self._minimum_face_pixels
                or bottom - top < self._minimum_face_pixels
            ):
                continue
            raw_embedding = tuple(float(value) for value in cast(Any, face).embedding.tolist())
            if _embedding_norm(raw_embedding) < self._minimum_embedding_norm:
                continue
            bbox = (
                min(1.0, max(0.0, left / width)),
                min(1.0, max(0.0, top / height)),
                min(1.0, max(0.0, right / width)),
                min(1.0, max(0.0, bottom / height)),
            )
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            samples.append(
                FaceEmbeddingSample(
                    sample_id=f"face-{timestamp_ms:012d}-{face_index:02d}",
                    start_ms=timestamp_ms,
                    end_ms=timestamp_ms + duration_ms,
                    confidence=confidence,
                    visual_bbox_xyxy=bbox,
                    embedding=_normalized_embedding(raw_embedding),
                )
            )
        return tuple(samples)

    def encode_video(
        self,
        media_path: Path,
        *,
        samples_per_second: float | None = None,
        maximum_samples: int = 256,
    ) -> tuple[FaceEmbeddingSample, ...]:
        """Decode with OpenCV and preserve timestamps and normalized spatial anchors."""
        samples_per_second = _face_sampling_rate(
            samples_per_second,
            self.execution_providers,
            maximum_samples,
        )
        path = media_path.resolve(strict=True)
        cv2 = cast(Any, _load_opencv())
        interval_ms = max(1, round(1_000 / samples_per_second))
        samples: list[FaceEmbeddingSample] = []
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ModelOutputError("OpenCV could not open the identity video")
        try:
            frame_rate = float(capture.get(cv2.CAP_PROP_FPS) or 30)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_ms = round(frame_count / frame_rate * 1_000) if frame_count > 0 else None
            next_sample_ms = 0
            frame_index = 0
            while True:
                available, image = capture.read()
                if not available:
                    break
                timestamp_ms = round(
                    float(capture.get(cv2.CAP_PROP_POS_MSEC)) or frame_index / frame_rate * 1_000
                )
                frame_index += 1
                if timestamp_ms < next_sample_ms:
                    continue
                next_sample_ms = timestamp_ms + interval_ms
                sample_duration_ms = (
                    max(1, min(interval_ms, duration_ms - timestamp_ms))
                    if duration_ms is not None and timestamp_ms < duration_ms
                    else interval_ms
                )
                samples.extend(
                    self.encode_frame(
                        image,
                        timestamp_ms=timestamp_ms,
                        duration_ms=sample_duration_ms,
                    )
                )
                if len(samples) >= maximum_samples:
                    return tuple(samples[:maximum_samples])
        finally:
            capture.release()
        return tuple(samples)


def recognize_faces_in_video(
    encoder: InsightFaceVideoEncoder,
    memory: SQLiteIdentityMemory,
    media_path: Path,
    *,
    tenant_id: str,
    observation_id: str,
    minimum_similarity: float,
    minimum_margin: float,
    samples_per_second: float | None = None,
) -> tuple[IdentityObservationInput, ...]:
    """Turn actual face detections into cloud-safe, spatially grounded identity intervals."""
    observations = []
    for sample in encoder.encode_video(media_path, samples_per_second=samples_per_second):
        match = memory.recognize_and_remember(
            LocalIdentitySample(
                tenant_id=tenant_id,
                kind=IdentityKind.FACE,
                source_observation_id=observation_id,
                sample_id=derive_stable_id("face_sample", observation_id, sample.sample_id),
                embedding=sample.embedding,
                model_reference=INSIGHTFACE_MODEL,
            ),
            minimum_similarity=minimum_similarity,
            minimum_margin=minimum_margin,
        )
        observations.append(
            match.to_observation_input(
                start_ms=sample.start_ms,
                end_ms=sample.end_ms,
                visual_bbox_xyxy=sample.visual_bbox_xyxy,
            ).model_copy(update={"confidence": min(sample.confidence, match.confidence)})
        )
    return tuple(observations)


def recognize_speakers(
    memory: SQLiteIdentityMemory,
    segments: tuple[SpeechSegment, ...],
    speaker_embeddings: tuple[SpeakerEmbeddingSample, ...],
    *,
    tenant_id: str,
    observation_id: str,
    minimum_similarity: float,
    minimum_margin: float,
    minimum_enrollment_duration_ms: int = 1_000,
    minimum_enrollment_confidence: float = 0.6,
) -> tuple[IdentityObservationInput, ...]:
    """Enroll one FunASR centroid per reliable speaker and retain every timed transcript."""
    if minimum_enrollment_duration_ms <= 0 or not (
        math.isfinite(minimum_enrollment_confidence) and 0.0 <= minimum_enrollment_confidence <= 1.0
    ):
        raise ValueError("voice enrollment thresholds are invalid")
    embeddings = {sample.speaker_label: sample.embedding for sample in speaker_embeddings}
    if len(embeddings) != len(speaker_embeddings):
        raise ValueError("speaker embedding labels must be unique")
    matches = {}
    for speaker_label, embedding in embeddings.items():
        turns = tuple(segment for segment in segments if segment.speaker_label == speaker_label)
        duration_ms = sum(segment.end_ms - segment.start_ms for segment in turns)
        if (
            duration_ms < minimum_enrollment_duration_ms
            or not turns
            or min(segment.confidence for segment in turns) < minimum_enrollment_confidence
        ):
            continue
        matches[speaker_label] = memory.recognize_and_remember(
            LocalIdentitySample(
                tenant_id=tenant_id,
                kind=IdentityKind.VOICE,
                source_observation_id=observation_id,
                sample_id=derive_stable_id("voice_sample", observation_id, speaker_label),
                embedding=embedding,
                model_reference=ERES2NETV2_MODEL,
            ),
            minimum_similarity=minimum_similarity,
            minimum_margin=minimum_margin,
        )
    observations = []
    for segment in segments:
        match = matches.get(segment.speaker_label) if segment.speaker_label is not None else None
        if match is None:
            observations.append(
                IdentityObservationInput(
                    identity_id=derive_stable_id(
                        "voice_observation",
                        tenant_id,
                        observation_id,
                        segment.speaker_label or segment.sample_id,
                    ),
                    kind=IdentityKind.VOICE,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    confidence=segment.confidence,
                    model_id=OBSERVATION_SCOPED_VOICE_MODEL.model_id,
                    model_revision=OBSERVATION_SCOPED_VOICE_MODEL.revision,
                    scope=IdentityScope.OBSERVATION,
                    transcript=segment.transcript,
                )
            )
            continue
        observations.append(
            match.to_observation_input(
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                transcript=segment.transcript,
            ).model_copy(update={"confidence": min(segment.confidence, match.confidence)})
        )
    return tuple(observations)


def _normalized_embedding(values: tuple[float, ...]) -> tuple[float, ...]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ModelOutputError("identity model returned a non-finite embedding")
    magnitude = _embedding_norm(values)
    if magnitude == 0.0:
        raise ModelOutputError("identity model returned a zero embedding")
    return tuple(value / magnitude for value in values)


def _embedding_norm(values: tuple[float, ...]) -> float:
    return math.sqrt(math.fsum(value * value for value in values))


def _face_sampling_rate(
    requested: float | None,
    providers: tuple[str, ...],
    maximum_samples: int,
) -> float:
    rate = requested if requested is not None else (1.0 if providers[0].startswith("CPU") else 5.0)
    if rate <= 0 or maximum_samples <= 0:
        raise ValueError("face sampling limits must be positive")
    return rate


def _load_opencv() -> ModuleType:
    try:
        return import_module("cv2")
    except ImportError as error:
        raise ModelUnavailableError("install OpenCV for video identity inference") from error


def _applied_face_providers(face_analysis: _FaceAnalysis) -> tuple[str, ...]:
    models = cast(Any, face_analysis).models.values()
    providers = tuple(
        dict.fromkeys(
            provider
            for model in models
            for provider in cast(list[str], model.session.get_providers())
        )
    )
    if not providers:
        raise ModelUnavailableError("InsightFace did not create an ONNX execution session")
    return providers
