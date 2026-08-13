"""Checks the local model-to-encrypted-identity handoff."""

from pathlib import Path
from typing import cast

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from mindbridge.core import IdentityScope
from mindbridge.edge.identity import SQLiteIdentityMemory
from mindbridge.edge.identity_inference import (
    FaceEmbeddingSample,
    InsightFaceVideoEncoder,
    SpeakerEmbeddingSample,
    SpeechSegment,
    recognize_faces_in_video,
    recognize_speakers,
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


def test_face_encoder_accepts_one_native_stream_frame() -> None:
    class Values:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        def tolist(self) -> list[float]:
            return self._values

    class Face:
        det_score = 0.95
        bbox = Values([10.0, 20.0, 60.0, 90.0])
        embedding = Values([30.0, 0.0])

    class Analysis:
        def prepare(self, *, ctx_id: int, det_size: tuple[int, int]) -> None:
            pass

        def get(self, _image: object) -> list[object]:
            return [Face()]

    class Image:
        shape = (100, 100, 3)

    encoder = InsightFaceVideoEncoder(
        Analysis(),
        execution_providers=("CUDAExecutionProvider",),
    )
    samples = encoder.encode_frame(Image(), timestamp_ms=1_200, duration_ms=200)

    assert len(samples) == 1
    assert (samples[0].start_ms, samples[0].end_ms) == (1_200, 1_400)
    assert samples[0].visual_bbox_xyxy == (0.1, 0.2, 0.6, 0.9)


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
    memory = SQLiteIdentityMemory(
        tmp_path / "edge.sqlite3",
        device_id="robot_01",
        encryption_key=AESGCM.generate_key(bit_length=256),
    )
    observations = recognize_speakers(
        memory,
        (
            SpeechSegment(
                sample_id="speech_01",
                start_ms=100,
                end_ms=1_100,
                transcript="请把红色工具递给我。",
                speaker_label="speaker_0",
            ),
        ),
        (SpeakerEmbeddingSample("speaker_0", (1.0, 0.0)),),
        tenant_id="tenant_01",
        observation_id="observation_01",
        minimum_similarity=0.8,
        minimum_margin=0.05,
    )

    assert observations[0].transcript == "请把红色工具递给我。"
    assert observations[0].scope is IdentityScope.DEVICE
    assert "embedding" not in observations[0].model_dump()


def test_asr_only_segment_is_observation_scoped_without_enrollment(tmp_path: Path) -> None:
    memory = SQLiteIdentityMemory(
        tmp_path / "edge.sqlite3",
        device_id="robot_01",
        encryption_key=AESGCM.generate_key(bit_length=256),
    )
    observations = recognize_speakers(
        memory,
        (
            SpeechSegment(
                sample_id="asr_only",
                start_ms=100,
                end_ms=2_100,
                confidence=0.95,
                transcript="Only VAD boundaries are known.",
            ),
        ),
        (),
        tenant_id="tenant_01",
        observation_id="observation_01",
        minimum_similarity=0.8,
        minimum_margin=0.05,
    )

    assert observations[0].scope is IdentityScope.OBSERVATION


def test_short_voice_turn_is_observation_scoped_without_enrollment(tmp_path: Path) -> None:
    memory = SQLiteIdentityMemory(
        tmp_path / "edge.sqlite3",
        device_id="robot_01",
        encryption_key=AESGCM.generate_key(bit_length=256),
    )
    observations = recognize_speakers(
        memory,
        (
            SpeechSegment(
                sample_id="speech_weak",
                start_ms=100,
                end_ms=700,
                confidence=0.9,
                transcript="嗯。",
                speaker_label="speaker_0",
            ),
        ),
        (SpeakerEmbeddingSample("speaker_0", (1.0, 0.0)),),
        tenant_id="tenant_01",
        observation_id="observation_01",
        minimum_similarity=0.8,
        minimum_margin=0.05,
    )

    assert observations[0].scope is IdentityScope.OBSERVATION
    assert observations[0].transcript == "嗯。"
    assert memory.forget_observation("tenant_01", "observation_01") == 0
