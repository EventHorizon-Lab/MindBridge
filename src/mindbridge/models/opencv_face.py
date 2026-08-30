"""Local YuNet detection and SFace identity exemplars through OpenCV."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Sequence
from importlib import import_module
from numbers import Real
from pathlib import Path
from threading import RLock
from typing import Protocol, cast

from mindbridge._telemetry import mark_model_requests, record_unmetered_model_usage
from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.models.base import FaceAnalysis, FaceEmbedding
from mindbridge.types import AssetRef, Modality


class _Image(Protocol):
    shape: tuple[int, ...]


class _Matrix(Protocol):
    def __iter__(self) -> Iterator[object]: ...

    def reshape(self, *shape: int) -> _Matrix: ...

    def tolist(self) -> object: ...


class _FaceDetector(Protocol):
    def setInputSize(self, size: tuple[int, int]) -> None: ...

    def detect(self, image: _Image) -> tuple[object, _Matrix | None]: ...


class _FaceRecognizer(Protocol):
    def alignCrop(self, image: _Image, face: object) -> _Image: ...

    def feature(self, image: _Image) -> _Matrix: ...


class _Capture(Protocol):
    def isOpened(self) -> bool: ...

    def get(self, property_id: int) -> float: ...

    def set(self, property_id: int, value: float) -> bool: ...

    def read(self) -> tuple[bool, _Image | None]: ...

    def release(self) -> None: ...


class _Factory(Protocol):
    def create(self, *args: object) -> object: ...


class _OpenCV(Protocol):
    FaceDetectorYN: _Factory
    FaceRecognizerSF: _Factory
    CAP_PROP_FPS: int
    CAP_PROP_FRAME_COUNT: int
    CAP_PROP_POS_MSEC: int

    def imread(self, path: str) -> _Image | None: ...

    def VideoCapture(self, path: str) -> _Capture: ...


class OpenCVFaceAnalyzer:
    """Extract normalized face exemplars locally with explicit YuNet and SFace weights."""

    def __init__(
        self,
        detector_model: str | Path,
        recognizer_model: str | Path,
        *,
        score_threshold: float = 0.9,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        frame_interval_ms: int = 1000,
        max_video_frames: int = 300,
    ) -> None:
        self._detector_model = _model_path(detector_model, "detector_model")
        self._recognizer_model = _model_path(recognizer_model, "recognizer_model")
        self._score_threshold = _unit_interval(score_threshold, "score_threshold")
        self._nms_threshold = _unit_interval(nms_threshold, "nms_threshold")
        self._top_k = _positive_integer(top_k, "top_k")
        self._frame_interval_ms = _positive_integer(frame_interval_ms, "frame_interval_ms")
        self._max_video_frames = _positive_integer(max_video_frames, "max_video_frames")
        detector_digest = _file_digest(self._detector_model)
        recognizer_digest = _file_digest(self._recognizer_model)
        self._model = f"opencv-sface:{recognizer_digest[:16]}"
        self._space = f"opencv-sface-v1:{recognizer_digest[:16]}"
        analysis = json.dumps(
            {
                "detector": detector_digest,
                "frame_interval_ms": self._frame_interval_ms,
                "max_video_frames": self._max_video_frames,
                "nms_threshold": self._nms_threshold,
                "score_threshold": self._score_threshold,
                "top_k": self._top_k,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self._analysis_space = (
            f"opencv-yunet-sface-v1:{hashlib.sha256(analysis.encode()).hexdigest()[:16]}"
        )
        self._cv: _OpenCV | None = None
        self._detector: _FaceDetector | None = None
        self._recognizer: _FaceRecognizer | None = None
        # OpenCV DNN objects mutate input shape and are not documented as thread-safe.
        self._lock = RLock()
        self._closed = False

    @property
    def face_capabilities(self) -> frozenset[Modality]:
        return frozenset({Modality.IMAGE, Modality.VIDEO})

    @property
    def face_model(self) -> str:
        return self._model

    @property
    def face_space(self) -> str:
        return self._space

    @property
    def face_analysis_space(self) -> str:
        return self._analysis_space

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[FaceAnalysis, ...]:
        if isinstance(assets, (str, bytes)):
            raise ValidationError("assets must contain resolved image or video AssetRef values")
        batch = tuple(assets)
        if any(
            not isinstance(asset, AssetRef)
            or not asset.is_resolved
            or asset.modality not in self.face_capabilities
            for asset in batch
        ):
            raise ValidationError("assets must contain resolved image or video AssetRef values")
        if not batch:
            return ()
        mark_model_requests(len(batch), token_usage_expected=0)
        with self._lock:
            if self._closed:
                raise ModelError("face backend is closed")
            cv, detector, recognizer = self._load()
            analyses = tuple(
                self._analyze_asset(cv, detector, recognizer, asset) for asset in batch
            )
        record_unmetered_model_usage(request_count=len(batch))
        return analyses

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._detector = None
            self._recognizer = None
            self._cv = None

    def _load(self) -> tuple[_OpenCV, _FaceDetector, _FaceRecognizer]:
        if self._cv is not None and self._detector is not None and self._recognizer is not None:
            return self._cv, self._detector, self._recognizer
        try:
            cv = cast(_OpenCV, import_module("cv2"))
            detector = cast(
                _FaceDetector,
                cv.FaceDetectorYN.create(
                    str(self._detector_model),
                    "",
                    (320, 320),
                    self._score_threshold,
                    self._nms_threshold,
                    self._top_k,
                ),
            )
            recognizer = cast(
                _FaceRecognizer,
                cv.FaceRecognizerSF.create(str(self._recognizer_model), ""),
            )
        except ImportError:
            raise ModelError(
                "OpenCV is unavailable; install MindBridge with the face extra"
            ) from None
        except Exception:
            raise ModelError("failed to load the OpenCV face recipe") from None
        self._cv, self._detector, self._recognizer = cv, detector, recognizer
        return cv, detector, recognizer

    def _analyze_asset(
        self,
        cv: _OpenCV,
        detector: _FaceDetector,
        recognizer: _FaceRecognizer,
        asset: AssetRef,
    ) -> FaceAnalysis:
        faces: list[FaceEmbedding] = []
        try:
            for observed_at_ms, image in self._frames(cv, asset):
                height, width = _image_size(image)
                detector.setInputSize((width, height))
                detected = detector.detect(image)[1]
                if detected is None:
                    continue
                for row in detected:
                    aligned = recognizer.alignCrop(image, row)
                    values = _feature_vector(recognizer.feature(aligned))
                    faces.append(
                        FaceEmbedding(
                            face_label=f"face_{len(faces)}",
                            values=values,
                            bounding_box=_bounding_box(row, width=width, height=height),
                            observed_at_ms=observed_at_ms,
                        )
                    )
        except ModelError:
            raise
        except Exception:
            raise ModelError("OpenCV face analysis failed") from None
        return FaceAnalysis(tuple(faces))

    def _frames(self, cv: _OpenCV, asset: AssetRef) -> Iterator[tuple[int | None, _Image]]:
        path = asset.path
        if path is None:
            raise ValidationError("face asset path is missing")
        if asset.modality is Modality.IMAGE:
            image = cv.imread(str(path))
            if image is None:
                raise ModelError("OpenCV could not decode the image")
            yield None, image
            return
        capture = cv.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ModelError("OpenCV could not open the video")
            fps = capture.get(cv.CAP_PROP_FPS)
            frame_count = capture.get(cv.CAP_PROP_FRAME_COUNT)
            if (
                not math.isfinite(fps)
                or not math.isfinite(frame_count)
                or fps <= 0
                or frame_count <= 0
            ):
                raise ModelError("OpenCV returned invalid video timing")
            duration_ms = max(0, round((frame_count - 1) * 1000 / fps))
            sample_count = min(
                self._max_video_frames,
                duration_ms // self._frame_interval_ms + 1,
            )
            times: tuple[int, ...]
            if sample_count == 1:
                times = (0,)
            else:
                times = tuple(
                    round(index * duration_ms / (sample_count - 1)) for index in range(sample_count)
                )
            decoded = False
            for observed_at_ms in times:
                capture.set(cv.CAP_PROP_POS_MSEC, float(observed_at_ms))
                ok, image = capture.read()
                if ok and image is not None:
                    decoded = True
                    yield observed_at_ms, image
            if not decoded:
                raise ModelError("OpenCV could not decode any sampled video frame")
        finally:
            capture.release()


def _model_path(value: str | Path, name: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, TypeError):
        raise ValidationError(f"{name} must be an existing model file") from None
    if not path.is_file():
        raise ValidationError(f"{name} must be an existing model file")
    return path


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        raise ValidationError("face model file could not be read") from None
    return digest.hexdigest()


def _unit_interval(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not 0.0 <= value <= 1.0
    ):
        raise ValidationError(f"{name} must be between zero and one")
    return float(value)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return value


def _image_size(image: _Image) -> tuple[int, int]:
    shape = image.shape
    if len(shape) < 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ModelError("OpenCV returned an invalid image")
    return int(shape[0]), int(shape[1])


def _feature_vector(value: _Matrix) -> tuple[float, ...]:
    flattened = value.reshape(-1).tolist()
    if not isinstance(flattened, list) or not flattened:
        raise ModelError("OpenCV returned an invalid face embedding")
    if any(isinstance(item, bool) or not isinstance(item, int | float) for item in flattened):
        raise ModelError("OpenCV returned an invalid face embedding")
    numbers = cast(list[int | float], flattened)
    vector = tuple(float(item) for item in numbers)
    magnitude = math.sqrt(math.fsum(item * item for item in vector))
    if not math.isfinite(magnitude) or magnitude == 0.0:
        raise ModelError("OpenCV returned an invalid face embedding")
    return tuple(item / magnitude for item in vector)


def _bounding_box(
    value: object,
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    try:
        raw = tuple(cast(Sequence[object], value))
    except TypeError:
        raise ModelError("OpenCV returned an invalid face detection") from None
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in raw):
        raise ModelError("OpenCV returned an invalid face detection")
    row = tuple(float(item) for item in cast(tuple[Real, ...], raw))
    if len(row) < 4 or any(not math.isfinite(item) for item in row[:4]):
        raise ModelError("OpenCV returned an invalid face detection")
    left = min(float(width), max(0.0, row[0]))
    top = min(float(height), max(0.0, row[1]))
    right = min(float(width), max(left, row[0] + row[2]))
    bottom = min(float(height), max(top, row[1] + row[3]))
    if right <= left or bottom <= top:
        raise ModelError("OpenCV returned an empty face detection")
    return left / width, top / height, (right - left) / width, (bottom - top) / height
