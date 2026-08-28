from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

import mindbridge.models.funasr as funasr_module
from mindbridge.exceptions import ModelError
from mindbridge.models.funasr import (
    DEFAULT_FUNASR_MODEL_ID,
    DEFAULT_FUNASR_MODEL_REVISION,
    DEFAULT_FUNASR_SPEAKER_MODEL_ID,
    DEFAULT_FUNASR_VAD_MODEL_ID,
    FunASRTranscriber,
)
from mindbridge.types import AssetRef, Modality


def test_default_funasr_recipe_transcribes_without_speakers_and_analyzes_them(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"fake wav")
    calls: list[dict[str, object]] = []

    class Pipeline:
        def __init__(self) -> None:
            self.spk_model: object | None = object()

        def generate(self, **kwargs: object) -> list[dict[str, object]]:
            calls.append({**kwargs, "speaker_enabled": self.spk_model is not None})
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

    def auto_model(**kwargs: object) -> Pipeline:
        arguments.update(kwargs)
        return Pipeline()

    def import_module(name: str) -> object:
        if name == "funasr":
            return SimpleNamespace(AutoModel=auto_model)
        if name == "torch":
            return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
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

    transcript = transcriber.transcribe((asset,))[0]
    first = transcriber.analyze((asset,))[0]
    second = transcriber.analyze((asset,))[0]

    assert transcript == "你好。"
    assert calls[0]["speaker_enabled"] is False
    assert calls[0]["return_spk_res"] is False
    assert calls[1]["speaker_enabled"] is True
    assert first == second
    assert first.turns[0].text == "你好。"
    assert first.turns[0].speaker_label == "0"
    assert first.speakers[0].values == pytest.approx((0.6, 0.8))
    assert arguments["model"] == DEFAULT_FUNASR_MODEL_ID
    assert arguments["model_revision"] == DEFAULT_FUNASR_MODEL_REVISION
    assert arguments["vad_model"] == DEFAULT_FUNASR_VAD_MODEL_ID
    assert arguments["spk_model"] == DEFAULT_FUNASR_SPEAKER_MODEL_ID
    assert arguments["device"] == "cpu"
    assert "trust_remote_code" not in arguments
    assert len(calls) == 3

    transcriber.close()
    with pytest.raises(ModelError, match="closed"):
        transcriber.analyze((asset,))


def test_silent_video_returns_empty_speech_without_loading_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "silent.mp4"
    video.write_bytes(b"fake silent video")
    opened: list[str] = []

    class Container:
        streams = SimpleNamespace(audio=())

        def close(self) -> None:
            pass

    def open_video(path: str) -> Container:
        opened.append(path)
        return Container()

    def import_module(name: str) -> object:
        if name == "av":
            return SimpleNamespace(open=open_video)
        raise AssertionError(f"silent video unexpectedly loaded {name}")

    monkeypatch.setattr(funasr_module, "import_module", import_module)
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
    transcriber = FunASRTranscriber()

    assert transcriber.transcribe((asset,)) == ("",)
    analysis = transcriber.analyze((asset,))[0]
    assert analysis.turns == ()
    assert analysis.speakers == ()
    assert opened == [str(video.resolve()), str(video.resolve())]


def test_vllm_batches_vad_spans_and_keeps_speaker_centroids(  # noqa: C901
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    label_calls: list[int] = []
    loaded: list[dict[str, object]] = []

    class Engine:
        def generate(self, **kwargs: object) -> list[dict[str, object]]:
            calls.append(kwargs)
            inputs = kwargs["inputs"]
            assert isinstance(inputs, list)
            return [{"text": f"turn {index}"} for index, _input in enumerate(inputs)]

    class Vad:
        def generate(self, **_kwargs: object) -> list[dict[str, object]]:
            return [{"value": [[0, 700]]}]

    class Speaker:
        def generate(self, **_kwargs: object) -> list[dict[str, object]]:
            raise AssertionError("speaker output is replaced by the focused CAM++ seam")

    def auto_model(**kwargs: object) -> object:
        loaded.append(kwargs)
        return Vad() if kwargs["model"] == DEFAULT_FUNASR_VAD_MODEL_ID else Speaker()

    def auto_model_vllm(**kwargs: object) -> Engine:
        loaded.append(kwargs)
        return Engine()

    def import_module(name: str) -> object:
        if name == "funasr":
            return SimpleNamespace(AutoModel=auto_model)
        if name == "funasr.auto.auto_model_vllm":
            return SimpleNamespace(AutoModelVLLM=auto_model_vllm)
        if name == "torch":
            return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        raise ImportError(name)

    def label_speakers(
        _pipeline: object,
        _audio: object,
        sentences: list[dict[str, object]],
    ) -> list[list[float]]:
        label_calls.append(len(sentences))
        for sentence in sentences:
            sentence["spk"] = 0
        return [[3.0, 4.0]]

    monkeypatch.setattr(funasr_module, "import_module", import_module)
    monkeypatch.setattr(funasr_module, "_load_waveform", lambda _asset: [0.0] * 12_000)
    monkeypatch.setattr(funasr_module, "_label_speakers", label_speakers)
    assets: list[AssetRef] = []
    for index in range(2):
        audio = tmp_path / f"voice-{index}.wav"
        audio.write_bytes(f"voice {index}".encode())
        digest = sha256(audio.read_bytes()).hexdigest()
        assets.append(
            AssetRef(
                digest,
                modality=Modality.AUDIO,
                media_type="audio/wav",
                size_bytes=audio.stat().st_size,
                sha256=digest,
                name=audio.name,
                path=audio,
            )
        )

    transcriber = FunASRTranscriber(engine="vllm", device="cuda")
    assert transcriber.space_id != FunASRTranscriber(engine="automodel", device="cuda").space_id
    transcripts = transcriber.transcribe(assets)
    analyses = transcriber.analyze(assets)

    assert transcripts == ("turn 0", "turn 1")
    assert [analysis.turns[0].text for analysis in analyses] == ["turn 0", "turn 1"]
    assert all(analysis.turns[0].speaker_label == "0" for analysis in analyses)
    assert all(analysis.speakers[0].values == pytest.approx((0.6, 0.8)) for analysis in analyses)
    assert label_calls == [1, 1]
    assert len(calls) == 2
    batch_inputs = calls[0]["inputs"]
    assert isinstance(batch_inputs, list) and len(batch_inputs) == 2
    assert calls[0]["repetition_penalty"] == 1.0
    assert loaded[0]["model"] == DEFAULT_FUNASR_MODEL_ID
    assert loaded[0]["dtype"] == "bf16"
    assert loaded[0]["gpu_memory_utilization"] == 0.5
    assert loaded[1]["model"] == DEFAULT_FUNASR_VAD_MODEL_ID
    assert loaded[2]["model"] == DEFAULT_FUNASR_SPEAKER_MODEL_ID
