from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

import mindbridge.models.funasr as funasr_module
from mindbridge.exceptions import ModelError
from mindbridge.models.base import SpeechBackend
from mindbridge.models.funasr import (
    DEFAULT_FUNASR_MODEL_ID,
    DEFAULT_FUNASR_MODEL_REVISION,
    DEFAULT_FUNASR_SPEAKER_MODEL_ID,
    DEFAULT_FUNASR_VAD_MODEL_ID,
    FunASRTranscriber,
)
from mindbridge.types import AssetRef, Modality


def test_funasr_delegates_execution_to_official_automodel_and_maps_speech(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"fake wav")
    calls: list[dict[str, object]] = []

    class _Pipeline:
        def generate(self, **kwargs: object) -> list[dict[str, object]]:
            calls.append(kwargs)
            return [
                {
                    "text": "你好。",
                    "sentence_info": [
                        {"start": 0, "end": 700, "text": ["你", "好", "。"], "spk": 0},
                        {"start": 700, "end": 900, "text": "<|Speech|>", "spk": 0},
                    ],
                    "spk_embedding_center": [[3.0, 4.0]],
                }
            ]

    arguments: dict[str, object] = {}

    def auto_model(**kwargs: object) -> _Pipeline:
        arguments.update(kwargs)
        return _Pipeline()

    def import_module(name: str) -> object:
        if name == "funasr":
            return SimpleNamespace(AutoModel=auto_model)
        raise ImportError(name)

    monkeypatch.setattr(funasr_module, "import_module", import_module)
    transcriber = FunASRTranscriber()
    digest = sha256(audio.read_bytes()).hexdigest()
    asset = AssetRef(
        digest,
        modality=Modality.AUDIO,
        media_type="audio/wav",
        size_bytes=audio.stat().st_size,
        sha256=digest,
        name=audio.name,
        path=audio,
    )

    first = transcriber.analyze((asset,))[0]
    second = transcriber.analyze((asset,))[0]

    assert isinstance(transcriber, SpeechBackend)
    assert first == second
    assert first.turns[0].text == "你好。"
    assert first.turns[0].speaker_label == "0"
    assert first.speakers[0].values == pytest.approx((0.6, 0.8))
    assert arguments["model"] == DEFAULT_FUNASR_MODEL_ID
    assert arguments["model_revision"] == DEFAULT_FUNASR_MODEL_REVISION
    assert arguments["vad_model"] == DEFAULT_FUNASR_VAD_MODEL_ID
    assert arguments["spk_model"] == DEFAULT_FUNASR_SPEAKER_MODEL_ID
    assert arguments["spk_mode"] == "vad_segment"
    expected_space = f"{DEFAULT_FUNASR_MODEL_ID}:speech:b47a175e755dea61"
    assert transcriber.transcription_space == expected_space
    assert "device" not in arguments
    assert "trust_remote_code" not in arguments
    assert len(calls) == 2

    transcriber.close()
    with pytest.raises(ModelError, match="closed"):
        transcriber.analyze((asset,))


def test_funasr_maps_an_empty_vad_result_to_silence(tmp_path: Path) -> None:
    video = tmp_path / "silent.mp4"
    video.write_bytes(b"fake video")

    class _Pipeline:
        def generate(self, **_kwargs: object) -> list[dict[str, object]]:
            return []

    digest = sha256(video.read_bytes()).hexdigest()
    asset = AssetRef(
        digest,
        modality=Modality.VIDEO,
        media_type="video/mp4",
        size_bytes=video.stat().st_size,
        sha256=digest,
        name=video.name,
        path=video,
    )

    assert FunASRTranscriber._analyze_one(_Pipeline(), asset).turns == ()


def test_funasr_sends_multiple_assets_in_one_official_batch(tmp_path: Path) -> None:
    paths = (tmp_path / "first.wav", tmp_path / "second.wav")
    for path in paths:
        path.write_bytes(b"fake wav")
    calls: list[object] = []

    class _Pipeline:
        def generate(self, **kwargs: object) -> list[dict[str, object]]:
            calls.append(kwargs["input"])
            return [
                {"text": "", "sentence_info": [], "spk_embedding_center": []},
                {"text": "", "sentence_info": [], "spk_embedding_center": []},
            ]

    assets = tuple(
        AssetRef(
            (digest := sha256(path.read_bytes()).hexdigest()),
            modality=Modality.AUDIO,
            media_type="audio/wav",
            size_bytes=path.stat().st_size,
            sha256=digest,
            name=path.name,
            path=path,
        )
        for path in paths
    )

    analyses = FunASRTranscriber._analyze_many(_Pipeline(), assets)

    assert len(analyses) == 2
    assert calls == [[str(path.resolve()) for path in paths]]
