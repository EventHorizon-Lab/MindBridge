"""Thin InsightFace and ModelScope adapters for device-local identity inference."""

from __future__ import annotations

import math
import subprocess
import tempfile
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
from mindbridge.models.compute import select_onnx_providers, select_torch_device

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


class _SpeakerModel(Protocol):
    embedding_model: object

    def __call__(self, audio: object) -> object: ...


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
                height, width = image.shape[:2]
                for face_index, face in enumerate(self._face_analysis.get(image)):
                    confidence = float(cast(Any, face).det_score)
                    left, top, right, bottom = (
                        float(value) for value in cast(Any, face).bbox.tolist()
                    )
                    if (
                        confidence < self._minimum_detection_confidence
                        or right - left < self._minimum_face_pixels
                        or bottom - top < self._minimum_face_pixels
                    ):
                        continue
                    raw_embedding = tuple(
                        float(value) for value in cast(Any, face).embedding.tolist()
                    )
                    if _embedding_norm(raw_embedding) < self._minimum_embedding_norm:
                        continue
                    embedding = _normalized_embedding(raw_embedding)
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
                            end_ms=(
                                max(timestamp_ms, min(timestamp_ms + interval_ms, duration_ms))
                                if duration_ms is not None
                                else timestamp_ms + interval_ms
                            ),
                            confidence=confidence,
                            visual_bbox_xyxy=bbox,
                            embedding=embedding,
                        )
                    )
                    if len(samples) >= maximum_samples:
                        return tuple(samples)
        finally:
            capture.release()
        return tuple(samples)


class ERes2NetV2SpeakerEncoder:
    """Use ModelScope's official ERes2NetV2 model and expose its normalized embedding."""

    def __init__(self, model: _SpeakerModel, *, device: str) -> None:
        self._model = model
        self.device = device

    @classmethod
    def load(
        cls,
        *,
        device: str | None = None,
    ) -> ERes2NetV2SpeakerEncoder:
        try:
            snapshot_module = import_module("modelscope.hub.snapshot_download")
            model_module = import_module("modelscope.models.audio.sv.ERes2NetV2")
        except ImportError as error:
            raise ModelUnavailableError(
                "install ModelScope for voice identity inference"
            ) from error
        selected_device = select_torch_device(device)
        try:
            model_directory = snapshot_module.snapshot_download(
                ERES2NETV2_MODEL.model_id,
                revision=ERES2NETV2_MODEL.revision,
                local_files_only=True,
            )
            revision_marker = Path(model_directory, ".mv").read_text()
            if not revision_marker.startswith(f"Revision:{ERES2NETV2_MODEL.revision},"):
                raise ValueError("cached ERes2NetV2 revision does not match")
        except (OSError, ValueError):
            model_directory = snapshot_module.snapshot_download(
                ERES2NETV2_MODEL.model_id,
                revision=ERES2NETV2_MODEL.revision,
            )
        model = cast(Any, model_module).SpeakerVerificationERes2NetV2(
            model_directory,
            {
                "sample_rate": 16_000,
                "embed_dim": 192,
                "baseWidth": 26,
                "scale": 2,
                "expansion": 2,
            },
            device=selected_device,
            pretrained_model="pretrained_eres2netv2.ckpt",
        )
        actual_device = str(next(cast(Any, model).embedding_model.parameters()).device)
        if selected_device.startswith("cuda") and not actual_device.startswith("cuda"):
            raise ModelUnavailableError("ERes2NetV2 silently fell back from requested CUDA")
        return cls(cast(_SpeakerModel, model), device=selected_device)

    def encode_media_segment(
        self,
        media_path: Path,
        segment: SpeechSegment,
        *,
        timeout_seconds: float = 120.0,
    ) -> tuple[float, ...]:
        """Encode one turn while preserving the small public adapter surface."""
        return self.encode_media_segments(
            media_path,
            (segment,),
            timeout_seconds=timeout_seconds,
        )[0]

    def encode_media_segments(
        self,
        media_path: Path,
        segments: tuple[SpeechSegment, ...],
        *,
        timeout_seconds: float = 120.0,
    ) -> tuple[tuple[float, ...], ...]:
        """Decode one 16 kHz waveform and encode every requested turn locally."""
        if timeout_seconds <= 0:
            raise ValueError("audio extraction timeout must be positive")
        if not segments:
            return ()
        waveform, sample_rate = _decode_mono_waveform(
            media_path,
            timeout_seconds=timeout_seconds,
        )
        embeddings = []
        for segment in segments:
            start = round(segment.start_ms * sample_rate / 1_000)
            end = min(len(waveform), round(segment.end_ms * sample_rate / 1_000))
            if start >= len(waveform) or end <= start:
                raise ModelOutputError("speaker segment exceeds the decoded audio")
            embedding = cast(Any, self._model(waveform[start:end]))
            if tuple(embedding.shape) != (1, 192):
                raise ModelOutputError("ERes2NetV2 returned an invalid speaker embedding")
            values = embedding.detach().squeeze(0).cpu().tolist()
            embeddings.append(_normalized_embedding(tuple(float(value) for value in values)))
        return tuple(embeddings)


def _decode_mono_waveform(
    media_path: Path,
    *,
    timeout_seconds: float,
) -> tuple[Any, int]:
    path = media_path.resolve(strict=True)
    with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
        try:
            subprocess.run(
                (
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-y",
                    audio.name,
                ),
                check=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ModelOutputError("FFmpeg could not extract the speaker audio") from error
        try:
            soundfile = import_module("soundfile")
            waveform, sample_rate = soundfile.read(audio.name, dtype="float32")
        except (ImportError, OSError, ValueError) as error:
            raise ModelOutputError("could not decode the extracted speaker audio") from error
    if sample_rate != 16_000 or getattr(waveform, "ndim", None) != 1 or len(waveform) == 0:
        raise ModelOutputError("speaker audio must be non-empty 16 kHz mono")
    return waveform, sample_rate


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


def recognize_speakers_in_media(
    encoder: ERes2NetV2SpeakerEncoder,
    memory: SQLiteIdentityMemory,
    media_path: Path,
    segments: tuple[SpeechSegment, ...],
    *,
    tenant_id: str,
    observation_id: str,
    minimum_similarity: float,
    minimum_margin: float,
    minimum_enrollment_duration_ms: int = 1_000,
    minimum_enrollment_confidence: float = 0.6,
) -> tuple[IdentityObservationInput, ...]:
    """Encode reliable turns; keep weak speech usable without enrolling its voiceprint."""
    if minimum_enrollment_duration_ms <= 0 or not (
        math.isfinite(minimum_enrollment_confidence) and 0.0 <= minimum_enrollment_confidence <= 1.0
    ):
        raise ValueError("voice enrollment thresholds are invalid")
    enrollment_indices = tuple(
        index
        for index, segment in enumerate(segments)
        if segment.speaker_label is not None
        and segment.end_ms - segment.start_ms >= minimum_enrollment_duration_ms
        and segment.confidence >= minimum_enrollment_confidence
    )
    embeddings = dict(
        zip(
            enrollment_indices,
            (
                encoder.encode_media_segments(
                    media_path,
                    tuple(segments[index] for index in enrollment_indices),
                )
                if enrollment_indices
                else ()
            ),
            strict=True,
        )
    )
    observations = []
    for index, segment in enumerate(segments):
        if index not in embeddings:
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
        match = memory.recognize_and_remember(
            LocalIdentitySample(
                tenant_id=tenant_id,
                kind=IdentityKind.VOICE,
                source_observation_id=observation_id,
                sample_id=derive_stable_id("voice_sample", observation_id, segment.sample_id),
                embedding=embeddings[index],
                model_reference=ERES2NETV2_MODEL,
            ),
            minimum_similarity=minimum_similarity,
            minimum_margin=minimum_margin,
        )
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
