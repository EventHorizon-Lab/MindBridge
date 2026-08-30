from __future__ import annotations

from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import mindbridge.models.opencv_face as opencv_face
from mindbridge.exceptions import ModelError
from mindbridge.models.base import FaceBackend
from mindbridge.models.opencv_face import OpenCVFaceAnalyzer
from mindbridge.types import AssetRef, Modality


class _Image:
    shape = (100, 200, 3)


class _Rows:
    def __iter__(self) -> Iterator[object]:
        return iter((np.asarray((10.0, 20.0, 50.0, 40.0, *([0.0] * 11)), dtype=np.float32),))


class _Feature:
    def reshape(self, *_shape: int) -> _Feature:
        return self

    def tolist(self) -> object:
        return [3.0, 4.0]


def _asset(path: Path, modality: Modality = Modality.IMAGE) -> AssetRef:
    digest = sha256(path.read_bytes()).hexdigest()
    return AssetRef(
        digest,
        modality=modality,
        media_type="image/png" if modality is Modality.IMAGE else "video/mp4",
        size_bytes=path.stat().st_size,
        sha256=digest,
        name=path.name,
        path=path,
    )


def test_opencv_face_analyzer_uses_yunet_and_sface_without_exporting_media(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    detector_model = tmp_path / "yunet.onnx"
    recognizer_model = tmp_path / "sface.onnx"
    image_path = tmp_path / "face.png"
    detector_model.write_bytes(b"yunet")
    recognizer_model.write_bytes(b"sface")
    image_path.write_bytes(b"image")
    calls: list[tuple[str, tuple[object, ...]]] = []

    class _Detector:
        def setInputSize(self, size: tuple[int, int]) -> None:
            calls.append(("size", size))

        def detect(self, _image: object) -> tuple[int, _Rows]:
            return 1, _Rows()

    class _Recognizer:
        def alignCrop(self, image: object, _face: object) -> object:
            return image

        def feature(self, _image: object) -> _Feature:
            return _Feature()

    def detector(*args: object) -> _Detector:
        calls.append(("detector", args))
        return _Detector()

    def recognizer(*args: object) -> _Recognizer:
        calls.append(("recognizer", args))
        return _Recognizer()

    def image(path: str) -> _Image:
        calls.append(("imread", (path,)))
        return _Image()

    cv = SimpleNamespace(
        FaceDetectorYN=SimpleNamespace(create=detector),
        FaceRecognizerSF=SimpleNamespace(create=recognizer),
        imread=image,
        VideoCapture=lambda _path: pytest.fail("image analysis must not open a video"),
        CAP_PROP_FPS=1,
        CAP_PROP_FRAME_COUNT=2,
        CAP_PROP_POS_MSEC=3,
    )
    monkeypatch.setattr(opencv_face, "import_module", lambda name: cv if name == "cv2" else None)
    analyzer = OpenCVFaceAnalyzer(detector_model, recognizer_model)

    analysis = analyzer.analyze((_asset(image_path),))[0]

    assert isinstance(analyzer, FaceBackend)
    assert analyzer.face_capabilities == {Modality.IMAGE, Modality.VIDEO}
    assert analysis.faces[0].values == pytest.approx((0.6, 0.8))
    assert analysis.faces[0].bounding_box == pytest.approx((0.05, 0.2, 0.25, 0.4))
    assert analysis.faces[0].observed_at_ms is None
    assert calls[2] == ("imread", (str(image_path),))
    assert calls[3] == ("size", (200, 100))

    analyzer.close()
    with pytest.raises(ModelError, match="closed"):
        analyzer.analyze((_asset(image_path),))


def test_opencv_face_analyzer_maps_missing_optional_runtime_to_model_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    detector_model = tmp_path / "yunet.onnx"
    recognizer_model = tmp_path / "sface.onnx"
    image_path = tmp_path / "face.png"
    detector_model.write_bytes(b"yunet")
    recognizer_model.write_bytes(b"sface")
    image_path.write_bytes(b"image")

    def missing(_name: str) -> object:
        raise ImportError

    monkeypatch.setattr(opencv_face, "import_module", missing)
    analyzer = OpenCVFaceAnalyzer(detector_model, recognizer_model)

    with pytest.raises(ModelError, match="face extra"):
        analyzer.analyze((_asset(image_path),))


def test_opencv_face_analyzer_samples_the_whole_video_with_a_hard_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    detector_model = tmp_path / "yunet.onnx"
    recognizer_model = tmp_path / "sface.onnx"
    video_path = tmp_path / "face.mp4"
    detector_model.write_bytes(b"yunet")
    recognizer_model.write_bytes(b"sface")
    video_path.write_bytes(b"video")
    sampled: list[int] = []

    class _Detector:
        def setInputSize(self, _size: tuple[int, int]) -> None:
            pass

        def detect(self, _image: object) -> tuple[int, _Rows]:
            return 1, _Rows()

    class _Recognizer:
        def alignCrop(self, image: object, _face: object) -> object:
            return image

        def feature(self, _image: object) -> _Feature:
            return _Feature()

    class _Capture:
        released = False

        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            return 30.0 if property_id == 1 else 91.0

        def set(self, _property_id: int, value: float) -> bool:
            sampled.append(round(value))
            return True

        def read(self) -> tuple[bool, _Image]:
            return True, _Image()

        def release(self) -> None:
            self.released = True

    capture = _Capture()
    cv = SimpleNamespace(
        FaceDetectorYN=SimpleNamespace(create=lambda *_args: _Detector()),
        FaceRecognizerSF=SimpleNamespace(create=lambda *_args: _Recognizer()),
        imread=lambda _path: pytest.fail("video analysis must not decode an image"),
        VideoCapture=lambda _path: capture,
        CAP_PROP_FPS=1,
        CAP_PROP_FRAME_COUNT=2,
        CAP_PROP_POS_MSEC=3,
    )
    monkeypatch.setattr(opencv_face, "import_module", lambda _name: cv)
    analyzer = OpenCVFaceAnalyzer(
        detector_model,
        recognizer_model,
        frame_interval_ms=1000,
        max_video_frames=3,
    )

    analysis = analyzer.analyze((_asset(video_path, Modality.VIDEO),))[0]

    assert sampled == [0, 1500, 3000]
    assert [face.observed_at_ms for face in analysis.faces] == sampled
    assert capture.released is True


def test_opencv_face_analyzer_rejects_a_video_when_no_sample_decodes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    detector_model = tmp_path / "yunet.onnx"
    recognizer_model = tmp_path / "sface.onnx"
    video_path = tmp_path / "broken.mp4"
    detector_model.write_bytes(b"yunet")
    recognizer_model.write_bytes(b"sface")
    video_path.write_bytes(b"video")

    class _Capture:
        released = False

        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            return 30.0 if property_id == 1 else 31.0

        def set(self, _property_id: int, _value: float) -> bool:
            return True

        def read(self) -> tuple[bool, None]:
            return False, None

        def release(self) -> None:
            self.released = True

    capture = _Capture()
    cv = SimpleNamespace(
        FaceDetectorYN=SimpleNamespace(create=lambda *_args: object()),
        FaceRecognizerSF=SimpleNamespace(create=lambda *_args: object()),
        imread=lambda _path: pytest.fail("video analysis must not decode an image"),
        VideoCapture=lambda _path: capture,
        CAP_PROP_FPS=1,
        CAP_PROP_FRAME_COUNT=2,
        CAP_PROP_POS_MSEC=3,
    )
    monkeypatch.setattr(opencv_face, "import_module", lambda _name: cv)
    analyzer = OpenCVFaceAnalyzer(detector_model, recognizer_model)

    with pytest.raises(ModelError, match="decode any sampled video frame"):
        analyzer.analyze((_asset(video_path, Modality.VIDEO),))

    assert capture.released is True
