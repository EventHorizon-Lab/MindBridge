from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

import mindbridge.models.insightface as insightface_module
from mindbridge.exceptions import ModelError
from mindbridge.models.insightface import InsightFaceRecognizer
from mindbridge.types import AssetRef, Modality


class _Values:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class _Image:
    shape = (100, 100, 3)


class _Face:
    det_score = 0.98
    bbox = _Values([10.0, 20.0, 60.0, 90.0])
    embedding = _Values([30.0, 0.0])


class _Session:
    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]


class _Analysis:
    def __init__(self) -> None:
        self.models = {"detection": SimpleNamespace(session=_Session())}
        self.prepared: tuple[int, float, tuple[int, int] | None] | None = None
        self.calls = 0

    def prepare(
        self,
        *,
        ctx_id: int,
        det_thresh: float,
        det_size: tuple[int, int] | None,
    ) -> None:
        self.prepared = (ctx_id, det_thresh, det_size)

    def get(self, _image: object) -> list[object]:
        self.calls += 1
        return [_Face()] if self.calls in {1, 3} else []


class _Capture:
    def __init__(self) -> None:
        self.frames = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        if property_id == 1:
            return 1.0
        return float(max(0, self.frames - 1) * 1_000)

    def read(self) -> tuple[bool, object | None]:
        self.frames += 1
        return (True, _Image()) if self.frames <= 5 else (False, None)

    def release(self) -> None:
        self.released = True


def test_insightface_reuses_upstream_image_and_video_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    analysis = _Analysis()
    arguments: dict[str, object] = {}

    def face_analysis(**kwargs: object) -> _Analysis:
        arguments.update(kwargs)
        return analysis

    capture = _Capture()
    cv2 = SimpleNamespace(
        CAP_PROP_FPS=1,
        CAP_PROP_POS_MSEC=2,
        imread=lambda _path: _Image(),
        VideoCapture=lambda _path: capture,
    )

    def import_module(name: str) -> object:
        if name == "insightface.app":
            return SimpleNamespace(FaceAnalysis=face_analysis)
        if name == "onnxruntime":
            return SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
        if name == "cv2":
            return cv2
        raise ImportError(name)

    monkeypatch.setattr(insightface_module, "import_module", import_module)
    recognizer = InsightFaceRecognizer(samples_per_second=1.0, maximum_samples=2)
    image = _asset(tmp_path / "portrait.png", Modality.IMAGE, "image/png")
    video = _asset(tmp_path / "clip.mp4", Modality.VIDEO, "video/mp4")

    image_result, video_result = recognizer.analyze((image, video))

    assert image_result.detections[0].bbox_xyxy == (0.1, 0.2, 0.6, 0.9)
    assert image_result.faces[0].values == (1.0, 0.0)
    assert len(video_result.detections) == 1
    assert video_result.detections[0].start_ms == 1_000
    assert analysis.calls == 3
    assert capture.released is True
    assert analysis.prepared == (-1, 0.6, None)
    assert arguments["name"] == "buffalo_l"
    assert arguments["allowed_modules"] == ["detection", "recognition"]
    assert recognizer.execution_providers == ("CPUExecutionProvider",)


def test_explicit_unavailable_onnx_provider_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def import_module(name: str) -> object:
        if name == "insightface.app":
            return SimpleNamespace(FaceAnalysis=lambda **_kwargs: _Analysis())
        if name == "onnxruntime":
            return SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
        raise ImportError(name)

    monkeypatch.setattr(insightface_module, "import_module", import_module)
    recognizer = InsightFaceRecognizer(device="cuda")

    with pytest.raises(ModelError, match="not available"):
        recognizer.analyze((_asset(tmp_path / "portrait.png", Modality.IMAGE, "image/png"),))


def _asset(path: Path, modality: Modality, media_type: str) -> AssetRef:
    path.write_bytes(b"media")
    digest = sha256(path.read_bytes()).hexdigest()
    return AssetRef(
        digest,
        modality=modality,
        media_type=media_type,
        size_bytes=path.stat().st_size,
        sha256=digest,
        name=path.name,
        path=path,
    )
