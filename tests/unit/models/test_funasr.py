from __future__ import annotations

import sys
import wave
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

import mindbridge.models.funasr as funasr_module
from mindbridge.exceptions import ModelError
from mindbridge.models.base import SpeechBackend, SpeechTurn
from mindbridge.models.funasr import (
    DEFAULT_FUNASR_MODEL_ID,
    DEFAULT_FUNASR_MODEL_REVISION,
    DEFAULT_FUNASR_RECIPE,
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
    usage: list[dict[str, object]] = []

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
    monkeypatch.setattr(
        funasr_module,
        "record_unmetered_model_usage",
        lambda **kwargs: usage.append(kwargs),
    )
    monkeypatch.setattr(funasr_module, "media_duration_seconds", lambda *_args, **_kwargs: 0.7)
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
    assert "punc_model" not in arguments
    expected_space = f"{DEFAULT_FUNASR_MODEL_ID}:speech:b47a175e755dea61"
    assert transcriber.transcription_space == expected_space
    assert "device" not in arguments
    assert "trust_remote_code" not in arguments
    assert len(calls) == 2
    assert usage == [
        {"request_count": 1, "audio_seconds": 0.7},
        {"request_count": 1, "audio_seconds": 0.7},
    ]

    transcriber.close()
    with pytest.raises(ModelError, match="closed"):
        transcriber.analyze((asset,))


def test_funasr_usage_counts_full_media_duration_for_batch_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "voice.wav"
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\0\0" * 32_000)
    monkeypatch.setitem(sys.modules, "av", None)

    class _Pipeline:
        calls = 0

        def generate(self, **_kwargs: object) -> list[dict[str, object]]:
            self.calls += 1
            if self.calls == 1:
                return []
            return [
                {
                    "text": "hello",
                    "sentence_info": [{"start": 100, "end": 700, "text": "hello", "spk": 0}],
                    "spk_embedding_center": [[1.0]],
                }
            ]

    usage: list[dict[str, object]] = []
    monkeypatch.setattr(
        funasr_module,
        "record_unmetered_model_usage",
        lambda **kwargs: usage.append(kwargs),
    )
    transcriber = FunASRTranscriber()
    transcriber._pipeline = _Pipeline()
    digest = sha256(audio.read_bytes()).hexdigest()
    result = transcriber.analyze(
        (
            AssetRef(
                digest,
                modality=Modality.AUDIO,
                media_type="audio/wav",
                size_bytes=audio.stat().st_size,
                sha256=digest,
                name=audio.name,
                path=audio,
            ),
        )
    )[0]

    assert result.turns[-1].end_ms == 700
    assert usage == [{"request_count": 2, "audio_seconds": pytest.approx(4.0)}]


def test_funasr_counts_a_failed_fallback_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"fake wav")
    counts: list[int] = []

    class _Pipeline:
        calls = 0

        def generate(self, **_kwargs: object) -> list[dict[str, object]]:
            self.calls += 1
            if self.calls == 1:
                return []
            raise RuntimeError("fallback failed")

    monkeypatch.setattr(
        funasr_module,
        "mark_model_requests",
        lambda count, **_kwargs: counts.append(count),
    )
    monkeypatch.setattr(
        funasr_module,
        "current_model_request_count",
        lambda: counts[-1] if counts else 0,
    )
    transcriber = FunASRTranscriber()
    transcriber._pipeline = _Pipeline()
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

    with pytest.raises(ModelError, match="failed to analyze speech"):
        transcriber.analyze((asset,))

    assert counts == [0, 1, 2]


def test_funasr_supports_timestamp_only_transcription_without_speaker_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"fake wav")
    arguments: dict[str, object] = {}
    calls: list[dict[str, object]] = []

    class _Pipeline:
        def generate(self, **kwargs: object) -> list[dict[str, object]]:
            calls.append(kwargs)
            return [{"text": "hello.", "timestamp": [[100, 400], [500, 900]]}]

    def auto_model(**kwargs: object) -> _Pipeline:
        arguments.update(kwargs)
        return _Pipeline()

    monkeypatch.setattr(
        funasr_module,
        "import_module",
        lambda _name: SimpleNamespace(AutoModel=auto_model),
    )
    recipe = replace(
        DEFAULT_FUNASR_RECIPE,
        speaker_model=None,
        speaker_revision=None,
    )
    transcriber = FunASRTranscriber(recipe=recipe)
    digest = sha256(audio.read_bytes()).hexdigest()
    result = transcriber.analyze(
        (
            AssetRef(
                digest,
                modality=Modality.AUDIO,
                media_type="audio/wav",
                size_bytes=audio.stat().st_size,
                sha256=digest,
                path=audio,
            ),
        )
    )[0]

    assert "spk_model" not in arguments
    assert "spk_mode" not in arguments
    assert calls[0]["sentence_timestamp"] is False
    assert calls[0]["return_spk_res"] is False
    assert result.turns == (SpeechTurn(100, 900, "hello.", None),)
    assert result.speakers == ()


def test_funasr_splits_long_timestamp_only_transcripts_into_bounded_turns() -> None:
    text = " ".join(f"word{index}" for index in range(1_000))
    timestamps = [[index * 10, (index + 1) * 10] for index in range(1_000)]

    result = funasr_module._analysis(
        {"text": text, "timestamp": timestamps},
        speaker_analysis=False,
    )

    assert len(result.turns) > 1
    assert " ".join(turn.text for turn in result.turns) == text
    assert all(len(turn.text) <= 4_096 for turn in result.turns)
    assert result.turns[0].start_ms == 0
    assert result.turns[-1].end_ms == 10_000


def test_funasr_does_not_hide_missing_speaker_turns_with_timestamp_fallback() -> None:
    with pytest.raises(ModelError, match="timed speaker turns"):
        funasr_module._analysis(
            {
                "text": "hello",
                "timestamp": [[0, 500]],
                "sentence_info": [],
                "spk_embedding_center": [[1.0, 0.0]],
            },
            speaker_analysis=True,
        )


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

    analyses, model_calls = FunASRTranscriber._analyze_many(_Pipeline(), assets)

    assert len(analyses) == 2
    assert calls == [[str(path.resolve()) for path in paths]]
    # Telemetry counts AutoModel calls, so batching two assets must report one request.
    assert model_calls == 1


def test_funasr_counts_the_per_asset_fallback_calls(tmp_path: Path) -> None:
    paths = (tmp_path / "first.wav", tmp_path / "second.wav")
    for path in paths:
        path.write_bytes(b"fake wav")
    calls: list[object] = []

    class _Pipeline:
        def generate(self, **kwargs: object) -> list[dict[str, object]]:
            calls.append(kwargs["input"])
            # A structurally unusable batch reply forces the per-asset fallback.
            return [] if isinstance(kwargs["input"], list) else [{"text": ""}]

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

    analyses, model_calls = FunASRTranscriber._analyze_many(_Pipeline(), assets)

    assert len(analyses) == 2
    assert len(calls) == 3
    assert model_calls == 3


def test_funasr_aligns_partial_keyed_batch_without_repeating_successes(tmp_path: Path) -> None:
    paths = (tmp_path / "first.wav", tmp_path / "second.wav")
    for path in paths:
        path.write_bytes(b"fake wav")
    calls: list[object] = []

    class _Pipeline:
        def generate(self, **kwargs: object) -> list[dict[str, object]]:
            calls.append(kwargs["input"])
            return [{"key": "first", "text": ""}]

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

    analyses, model_calls = FunASRTranscriber._analyze_many(_Pipeline(), assets)

    assert tuple(analysis.turns for analysis in analyses) == ((), ())
    assert calls == [[str(path.resolve()) for path in paths]]
    assert model_calls == 1
