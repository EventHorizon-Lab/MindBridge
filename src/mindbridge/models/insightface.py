"""Lazy InsightFace SCRFD and ArcFace adapter for local face recognition."""

from __future__ import annotations

import math
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, cast

from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.models.base import FaceAnalysis, FaceDetection, FaceEmbedding
from mindbridge.types import AssetRef, Modality

DEFAULT_INSIGHTFACE_MODEL = "buffalo_l"
DEFAULT_INSIGHTFACE_MODEL_REVISION = "v0.7"


class _FaceAnalysis(Protocol):
    models: dict[str, object]

    def prepare(
        self,
        *,
        ctx_id: int,
        det_thresh: float,
        det_size: tuple[int, int] | None,
    ) -> None: ...

    def get(self, image: object) -> list[object]: ...


class _VideoCapture(Protocol):
    def isOpened(self) -> bool: ...

    def get(self, property_id: int) -> float: ...

    def read(self) -> tuple[bool, object]: ...

    def release(self) -> None: ...


class _OpenCV(Protocol):
    CAP_PROP_FPS: int
    CAP_PROP_POS_MSEC: int

    def imread(self, path: str) -> object | None: ...

    def VideoCapture(self, path: str) -> _VideoCapture: ...


class InsightFaceRecognizer:
    """Detect faces with SCRFD and encode them with ArcFace through ONNX Runtime."""

    capabilities = frozenset({Modality.IMAGE, Modality.VIDEO})

    def __init__(
        self,
        *,
        model: str = DEFAULT_INSIGHTFACE_MODEL,
        device: str = "auto",
        model_root: str | Path | None = None,
        detection_size: tuple[int, int] | None = None,
        minimum_detection_score: float = 0.6,
        minimum_embedding_norm: float = 20.0,
        minimum_face_pixels: int = 32,
        samples_per_second: float = 1.0,
        maximum_samples: int = 256,
    ) -> None:
        self._model = _text(model, "InsightFace model")
        self._device = _device(device)
        try:
            self._model_root = None if model_root is None else Path(model_root).expanduser()
        except TypeError:
            raise ValidationError("InsightFace model_root must be a filesystem path") from None
        self._detection_size = _detection_size(detection_size)
        self._minimum_detection_score = _unit_interval(
            minimum_detection_score,
            "minimum_detection_score",
        )
        if (
            isinstance(minimum_embedding_norm, bool)
            or not isinstance(minimum_embedding_norm, int | float)
            or not math.isfinite(float(minimum_embedding_norm))
            or minimum_embedding_norm <= 0
        ):
            raise ValidationError("minimum_embedding_norm must be positive")
        if isinstance(minimum_face_pixels, bool) or not isinstance(minimum_face_pixels, int):
            raise ValidationError("minimum_face_pixels must be a positive integer")
        if minimum_face_pixels <= 0:
            raise ValidationError("minimum_face_pixels must be a positive integer")
        if (
            isinstance(samples_per_second, bool)
            or not isinstance(samples_per_second, int | float)
            or not math.isfinite(float(samples_per_second))
            or samples_per_second <= 0
        ):
            raise ValidationError("samples_per_second must be positive")
        if isinstance(maximum_samples, bool) or not isinstance(maximum_samples, int):
            raise ValidationError("maximum_samples must be a positive integer")
        if maximum_samples <= 0:
            raise ValidationError("maximum_samples must be a positive integer")
        self._minimum_embedding_norm = float(minimum_embedding_norm)
        self._minimum_face_pixels = minimum_face_pixels
        self._samples_per_second = float(samples_per_second)
        self._maximum_samples = maximum_samples
        self._analysis: _FaceAnalysis | None = None
        self._providers: tuple[str, ...] = ()
        # ponytail: serialize one InsightFace session; use separate recognizers only after
        # measured throughput justifies concurrent model copies.
        self._lock = RLock()
        self._closed = False

    @property
    def model_id(self) -> str:
        return f"insightface/{self._model}@{DEFAULT_INSIGHTFACE_MODEL_REVISION}"

    @property
    def space_id(self) -> str:
        detection = (
            "auto"
            if self._detection_size is None
            else f"{self._detection_size[0]}x{self._detection_size[1]}"
        )
        return (
            f"{self.model_id}:arcface-l2:scrfd-{detection}:"
            f"det-{self._minimum_detection_score:g}:norm-{self._minimum_embedding_norm:g}:"
            f"face-{self._minimum_face_pixels}:fps-{self._samples_per_second:g}:"
            f"max-{self._maximum_samples}:v1"
        )

    @property
    def execution_providers(self) -> tuple[str, ...]:
        """Return providers actually loaded, or an empty tuple before first inference."""
        with self._lock:
            return self._providers

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[FaceAnalysis, ...]:
        supplied = tuple(assets)
        if any(
            not isinstance(asset, AssetRef)
            or not asset.is_resolved
            or asset.modality not in self.capabilities
            for asset in supplied
        ):
            raise ValidationError("InsightFace requires resolved image or video assets")
        with self._lock:
            if self._closed:
                raise ModelError("InsightFace recognizer is closed")
            analysis = self._load()
            try:
                cv2 = cast(_OpenCV, import_module("cv2"))
                return tuple(self._analyze_asset(asset, analysis, cv2) for asset in supplied)
            except (ModelError, ValidationError):
                raise
            except Exception as error:
                raise ModelError("InsightFace failed to analyze face input") from error

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._analysis = None
            self._providers = ()

    def _load(self) -> _FaceAnalysis:
        if self._analysis is not None:
            return self._analysis
        try:
            insightface = import_module("insightface.app")
            onnxruntime = cast(Any, import_module("onnxruntime"))
        except ImportError as error:
            raise ModelError("install mindbridge[face] to use local face recognition") from error
        providers = _select_providers(
            tuple(cast(list[str], onnxruntime.get_available_providers())),
            self._device,
        )
        preload_dlls = getattr(onnxruntime, "preload_dlls", None)
        if "CUDAExecutionProvider" in providers and callable(preload_dlls):
            preload_dlls()
        arguments: dict[str, object] = {
            "name": self._model,
            "allowed_modules": ["detection", "recognition"],
            "providers": list(providers),
        }
        if self._model_root is not None:
            arguments["root"] = str(self._model_root)
        try:
            analysis = cast(_FaceAnalysis, insightface.FaceAnalysis(**arguments))
            analysis.prepare(
                ctx_id=-1 if providers[0] == "CPUExecutionProvider" else 0,
                det_thresh=self._minimum_detection_score,
                det_size=self._detection_size,
            )
        except Exception as error:
            raise ModelError(f"failed to load InsightFace model {self._model!r}") from error
        applied = _applied_providers(analysis)
        required = {
            "cuda": "CUDAExecutionProvider",
            "tensorrt": "TensorrtExecutionProvider",
        }.get(self._device)
        if required is not None and required not in applied:
            raise ModelError(f"requested ONNX {self._device} provider failed to load")
        self._analysis = analysis
        self._providers = applied
        return analysis

    def _analyze_asset(
        self,
        asset: AssetRef,
        analysis: _FaceAnalysis,
        cv2: _OpenCV,
    ) -> FaceAnalysis:
        if asset.modality is Modality.IMAGE:
            image = cv2.imread(str(asset.path))
            if image is None:
                raise ModelError("OpenCV could not decode an image for face recognition")
            detections, embeddings = self._encode_frame(analysis, image, prefix="image")
            return FaceAnalysis(detections=detections, faces=embeddings)
        return self._analyze_video(cast(Path, asset.path), analysis, cv2)

    def _analyze_video(
        self,
        path: Path,
        analysis: _FaceAnalysis,
        cv2: _OpenCV,
    ) -> FaceAnalysis:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ModelError("OpenCV could not open a video for face recognition")
        detections: list[FaceDetection] = []
        embeddings: list[FaceEmbedding] = []
        interval_ms = max(1, round(1_000 / self._samples_per_second))
        next_sample_ms = 0
        frame_index = 0
        samples = 0
        try:
            frame_rate = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            while samples < self._maximum_samples:
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
                samples += 1
                found, vectors = self._encode_frame(
                    analysis,
                    image,
                    prefix=f"video-{frame_index:08d}",
                    start_ms=timestamp_ms,
                    end_ms=timestamp_ms + interval_ms,
                )
                detections.extend(found)
                embeddings.extend(vectors)
        finally:
            capture.release()
        return FaceAnalysis(detections=tuple(detections), faces=tuple(embeddings))

    def _encode_frame(
        self,
        analysis: _FaceAnalysis,
        image: object,
        *,
        prefix: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> tuple[tuple[FaceDetection, ...], tuple[FaceEmbedding, ...]]:
        try:
            height, width = cast(Any, image).shape[:2]
        except (AttributeError, TypeError, ValueError) as error:
            raise ModelError("InsightFace requires a non-empty BGR image") from error
        if height <= 0 or width <= 0:
            raise ModelError("InsightFace requires a non-empty BGR image")
        detections = []
        embeddings = []
        for index, face in enumerate(analysis.get(image)):
            encoded = self._face_value(face, width=width, height=height)
            if encoded is None:
                continue
            bbox, score, values = encoded
            label = f"{prefix}-{index:04d}"
            detections.append(
                FaceDetection(
                    face_label=label,
                    bbox_xyxy=bbox,
                    detection_score=score,
                    start_ms=start_ms,
                    end_ms=end_ms,
                )
            )
            embeddings.append(FaceEmbedding(label, values))
        return tuple(detections), tuple(embeddings)

    def _face_value(
        self,
        face: object,
        *,
        width: int,
        height: int,
    ) -> tuple[tuple[float, float, float, float], float, tuple[float, ...]] | None:
        try:
            score = float(cast(Any, face).det_score)
            left, top, right, bottom = (float(value) for value in cast(Any, face).bbox.tolist())
            raw = tuple(float(value) for value in cast(Any, face).embedding.tolist())
        except (AttributeError, TypeError, ValueError) as error:
            raise ModelError("InsightFace returned an invalid face") from error
        if any(not math.isfinite(value) for value in (score, left, top, right, bottom, *raw)):
            raise ModelError("InsightFace returned a non-finite face")
        if (
            score < self._minimum_detection_score
            or right - left < self._minimum_face_pixels
            or bottom - top < self._minimum_face_pixels
            or _norm(raw) < self._minimum_embedding_norm
        ):
            return None
        bbox = (
            min(1.0, max(0.0, left / width)),
            min(1.0, max(0.0, top / height)),
            min(1.0, max(0.0, right / width)),
            min(1.0, max(0.0, bottom / height)),
        )
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None
        return bbox, max(0.0, min(1.0, score)), _normalized(raw)


def _select_providers(available: tuple[str, ...], device: str) -> tuple[str, ...]:
    preferred = {
        "auto": ("CUDAExecutionProvider",),
        "cuda": ("CUDAExecutionProvider",),
        "tensorrt": ("TensorrtExecutionProvider", "CUDAExecutionProvider"),
        "cpu": (),
    }[device]
    required = {
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
    }.get(device)
    if required is not None and required not in available:
        raise ModelError(f"requested ONNX {device} provider is not available")
    selected = tuple(provider for provider in preferred if provider in available)
    if "CPUExecutionProvider" in available:
        selected += ("CPUExecutionProvider",)
    if not selected:
        raise ModelError("no supported ONNX execution provider is available")
    return selected


def _applied_providers(analysis: _FaceAnalysis) -> tuple[str, ...]:
    try:
        providers = tuple(
            dict.fromkeys(
                provider
                for model in analysis.models.values()
                for provider in cast(Any, model).session.get_providers()
            )
        )
    except (AttributeError, TypeError) as error:
        raise ModelError("InsightFace did not expose its ONNX execution providers") from error
    if not providers:
        raise ModelError("InsightFace did not create an ONNX execution session")
    return providers


def _normalized(values: tuple[float, ...]) -> tuple[float, ...]:
    magnitude = _norm(values)
    if not values or not math.isfinite(magnitude) or magnitude == 0:
        raise ModelError("InsightFace returned a zero or non-finite embedding")
    return tuple(value / magnitude for value in values)


def _norm(values: tuple[float, ...]) -> float:
    return math.sqrt(math.fsum(value * value for value in values))


def _device(value: object) -> str:
    normalized = _text(value, "InsightFace device").lower()
    if normalized not in {"auto", "cpu", "cuda", "tensorrt"}:
        raise ValidationError("InsightFace device must be auto, cpu, cuda, or tensorrt")
    return normalized


def _detection_size(value: object) -> tuple[int, int] | None:
    if value is None:
        return None
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise ValidationError("detection_size must be None or two positive integers")
    return value


def _unit_interval(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not 0.0 <= value <= 1.0
    ):
        raise ValidationError(f"{name} must be between zero and one")
    return float(value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be non-empty text")
    return value.strip()
