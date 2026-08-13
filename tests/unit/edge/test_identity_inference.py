"""Checks the local model-to-encrypted-identity handoff."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from mindbridge.core import IdentityScope
from mindbridge.edge import identity_inference
from mindbridge.edge.identity import SQLiteIdentityMemory
from mindbridge.edge.identity_inference import (
    ERES2NETV2_MODEL,
    ERes2NetV2SpeakerEncoder,
    FaceEmbeddingSample,
    InsightFaceVideoEncoder,
    SpeechSegment,
    recognize_faces_in_video,
    recognize_speakers_in_media,
)


class _StaticFaceEncoder:
    def encode_video(
        self,
        _media_path: Path,
        *,
        samples_per_second: float | None,
        maximum_samples: int = 256,
    ) -> tuple[FaceEmbeddingSample, ...]:
        assert samples_per_second is None
        assert maximum_samples == 256
        return (
            FaceEmbeddingSample(
                sample_id="face-000000000000-00",
                start_ms=0,
                end_ms=500,
                confidence=0.96,
                visual_bbox_xyxy=(0.1, 0.2, 0.4, 0.8),
                embedding=(1.0, 0.0),
            ),
            FaceEmbeddingSample(
                sample_id="face-000000000500-00",
                start_ms=500,
                end_ms=1_000,
                confidence=0.94,
                visual_bbox_xyxy=(0.11, 0.2, 0.41, 0.8),
                embedding=(0.99, 0.01),
            ),
        )


def test_speaker_encoder_loads_the_official_modelscope_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings: dict[str, object] = {}

    class Parameter:
        device = "cuda:0"

    class EmbeddingModel:
        def parameters(self) -> object:
            return iter((Parameter(),))

    class SpeakerModel:
        embedding_model = EmbeddingModel()

        def __init__(self, model_directory: str, config: object, **kwargs: object) -> None:
            settings.update(
                model_directory=model_directory,
                config=config,
                **kwargs,
            )

    model_directory = Path("/models/eres2netv2")

    def imported(name: str) -> object:
        if name == "modelscope.hub.snapshot_download":
            return SimpleNamespace(
                snapshot_download=lambda model_id, revision, **kwargs: (
                    settings.update(
                        model_id=model_id,
                        revision=revision,
                        snapshot_kwargs=kwargs,
                    )
                    or str(model_directory)
                )
            )
        return SimpleNamespace(SpeakerVerificationERes2NetV2=SpeakerModel)

    monkeypatch.setattr(Path, "read_text", lambda _path: "Revision:v1.0.1,CreatedAt:1")
    monkeypatch.setattr(identity_inference, "import_module", imported)
    monkeypatch.setattr(identity_inference, "select_torch_device", lambda _device: "cuda")

    encoder = ERes2NetV2SpeakerEncoder.load()

    assert encoder.device == "cuda"
    assert settings["model_id"] == ERES2NETV2_MODEL.model_id
    assert settings["revision"] == ERES2NETV2_MODEL.revision
    assert settings["snapshot_kwargs"] == {"local_files_only": True}
    assert settings["device"] == "cuda"


def test_speaker_encoder_decodes_once_for_multiple_turns(  # noqa: C901 - dependency-free fakes
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Waveform:
        ndim = 1

        def __init__(self) -> None:
            self.slices: list[slice] = []

        def __len__(self) -> int:
            return 32_000

        def __getitem__(self, item: slice) -> object:
            self.slices.append(item)
            return object()

    class Embedding:
        shape = (1, 192)

        def detach(self) -> "Embedding":
            return self

        def squeeze(self, _dimension: int) -> "Embedding":
            return self

        def cpu(self) -> "Embedding":
            return self

        def tolist(self) -> list[float]:
            return [1.0, *([0.0] * 191)]

    class Model:
        embedding_model = object()

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, _audio: object) -> Embedding:
            self.calls += 1
            return Embedding()

    waveform = Waveform()
    decode_calls = 0

    def decode(_path: Path, *, timeout_seconds: float) -> tuple[Waveform, int]:
        nonlocal decode_calls
        decode_calls += 1
        assert timeout_seconds == 120.0
        return waveform, 16_000

    monkeypatch.setattr(identity_inference, "_decode_mono_waveform", decode)
    model = Model()
    encoder = ERes2NetV2SpeakerEncoder(
        cast(identity_inference._SpeakerModel, model),
        device="cuda",
    )

    embeddings = encoder.encode_media_segments(
        tmp_path / "unused.wav",
        (
            SpeechSegment("speech_0", 0, 500, speaker_label="speaker_0"),
            SpeechSegment("speech_1", 1_000, 2_000, speaker_label="speaker_0"),
        ),
    )

    assert decode_calls == 1
    assert model.calls == 2
    assert waveform.slices == [slice(0, 8_000), slice(16_000, 32_000)]
    assert len(embeddings) == 2


def test_face_inference_retains_embeddings_locally_and_sends_spatial_ids(tmp_path: Path) -> None:
    memory = SQLiteIdentityMemory(
        tmp_path / "edge.sqlite3",
        device_id="robot_01",
        encryption_key=AESGCM.generate_key(bit_length=256),
    )

    observations = recognize_faces_in_video(
        cast(InsightFaceVideoEncoder, _StaticFaceEncoder()),
        memory,
        tmp_path / "unused.mp4",
        tenant_id="tenant_01",
        observation_id="observation_01",
        minimum_similarity=0.8,
        minimum_margin=0.05,
    )

    assert len(observations) == 2
    assert observations[0].identity_id == observations[1].identity_id
    assert observations[0].visual_bbox_xyxy == (0.1, 0.2, 0.4, 0.8)
    assert all("embedding" not in observation.model_dump() for observation in observations)


def test_voice_inference_retains_timed_transcript_but_not_embedding(tmp_path: Path) -> None:
    class Encoder:
        def encode_media_segments(
            self,
            _media_path: Path,
            segments: tuple[SpeechSegment, ...],
        ) -> tuple[tuple[float, ...], ...]:
            return tuple((1.0, 0.0) for _ in segments)

    memory = SQLiteIdentityMemory(
        tmp_path / "edge.sqlite3",
        device_id="robot_01",
        encryption_key=AESGCM.generate_key(bit_length=256),
    )
    observations = recognize_speakers_in_media(
        cast(ERes2NetV2SpeakerEncoder, Encoder()),
        memory,
        tmp_path / "unused.mp4",
        (
            SpeechSegment(
                sample_id="speech_01",
                start_ms=100,
                end_ms=1_100,
                transcript="请把红色工具递给我。",
                speaker_label="speaker_0",
            ),
        ),
        tenant_id="tenant_01",
        observation_id="observation_01",
        minimum_similarity=0.8,
        minimum_margin=0.05,
    )

    assert observations[0].transcript == "请把红色工具递给我。"
    assert observations[0].scope is IdentityScope.DEVICE
    assert "embedding" not in observations[0].model_dump()


def test_asr_only_segment_is_observation_scoped_without_enrollment(tmp_path: Path) -> None:
    class Encoder:
        def encode_media_segment(
            self,
            _media_path: Path,
            _segment: SpeechSegment,
        ) -> tuple[float, ...]:
            raise AssertionError("VAD alone must not create a biometric template")

    memory = SQLiteIdentityMemory(
        tmp_path / "edge.sqlite3",
        device_id="robot_01",
        encryption_key=AESGCM.generate_key(bit_length=256),
    )
    observations = recognize_speakers_in_media(
        cast(ERes2NetV2SpeakerEncoder, Encoder()),
        memory,
        tmp_path / "unused.mp4",
        (
            SpeechSegment(
                sample_id="asr_only",
                start_ms=100,
                end_ms=2_100,
                confidence=0.95,
                transcript="Only VAD boundaries are known.",
            ),
        ),
        tenant_id="tenant_01",
        observation_id="observation_01",
        minimum_similarity=0.8,
        minimum_margin=0.05,
    )

    assert observations[0].scope is IdentityScope.OBSERVATION


def test_short_voice_turn_is_observation_scoped_without_enrollment(tmp_path: Path) -> None:
    class Encoder:
        def encode_media_segment(
            self,
            _media_path: Path,
            _segment: SpeechSegment,
        ) -> tuple[float, ...]:
            raise AssertionError("weak speech must not create a biometric template")

    memory = SQLiteIdentityMemory(
        tmp_path / "edge.sqlite3",
        device_id="robot_01",
        encryption_key=AESGCM.generate_key(bit_length=256),
    )
    observations = recognize_speakers_in_media(
        cast(ERes2NetV2SpeakerEncoder, Encoder()),
        memory,
        tmp_path / "unused.mp4",
        (
            SpeechSegment(
                sample_id="speech_weak",
                start_ms=100,
                end_ms=700,
                confidence=0.9,
                transcript="嗯。",
            ),
        ),
        tenant_id="tenant_01",
        observation_id="observation_01",
        minimum_similarity=0.8,
        minimum_margin=0.05,
    )

    assert observations[0].scope is IdentityScope.OBSERVATION
    assert observations[0].transcript == "嗯。"
    assert memory.forget_observation("tenant_01", "observation_01") == 0
