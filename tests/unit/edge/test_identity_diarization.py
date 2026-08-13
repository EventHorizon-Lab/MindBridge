"""Contract check for OpenAI-SDK audiovisual speaker segmentation."""

import json
import shutil
import subprocess
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import cast

import httpx
import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from openai import AsyncOpenAI

from mindbridge.contracts import IdentityObservationInput
from mindbridge.core import IdentityKind, IdentityScope, ModelReference
from mindbridge.edge import identity_diarization
from mindbridge.edge.identity import FaceVoiceAssociationEvidence, SQLiteIdentityMemory
from mindbridge.edge.identity_diarization import (
    FunASRSpeechTranscriber,
    IdentityMatchingThresholds,
    NemoSortformerSpeechDiarizer,
    OpenAIAVSpeechSegmenter,
    OpenAIVisualActiveSpeakerMatcher,
    fuse_speech_segments,
    recognize_identities_in_av_segment,
)
from mindbridge.edge.identity_inference import (
    ERes2NetV2SpeakerEncoder,
    FaceEmbeddingSample,
    InsightFaceVideoEncoder,
    SpeechSegment,
)
from mindbridge.models.openai_omni import normalize_openai_base_url


async def test_diarizer_sends_native_av_and_returns_bounded_turns(
    tmp_path: Path,
) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"real-media-placeholder")

    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        messages = cast(list[dict[str, object]], payload["messages"])
        content = cast(list[dict[str, object]], messages[1]["content"])
        video = content[1]
        assert request.url.path == "/api/v1/chat/completions"
        assert payload["stream"] is True
        assert video["type"] == "video_url"
        assert cast(dict[str, str], video["video_url"])["url"].startswith("data:video/mp4;base64,")
        assert [part["type"] for part in content] == ["text", "video_url", "text"]
        return _streaming_response(
            {"segments": [{"start_ms": 100, "end_ms": 900, "transcript": "Hello."}]}
        )

    segmenter = _segmenter(respond)
    try:
        segments = await segmenter.segment_file(media, duration_ms=1_000)
    finally:
        await segmenter.close()

    assert [
        (item.start_ms, item.end_ms, item.confidence, item.transcript) for item in segments
    ] == [(100, 900, 0.8, "Hello.")]


async def test_funasr_splits_timestamped_speech_without_blocking_the_edge_loop(
    tmp_path: Path,
) -> None:
    class Pipeline:
        def generate(self, **_kwargs: object) -> list[dict[str, object]]:
            return [
                {
                    "text": "你 好 hello world",
                    "timestamp": [[0, 300], [300, 700], [1_300, 1_700], [1_700, 2_200]],
                }
            ]

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"placeholder")
    segments = await FunASRSpeechTranscriber(Pipeline(), device="cuda").segment_file(media)

    assert [(item.start_ms, item.end_ms, item.transcript) for item in segments] == [
        (0, 700, "你好"),
        (1_300, 2_200, "hello world"),
    ]


async def test_sortformer_keeps_upstream_speaker_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Probabilities:
        def detach(self) -> object:
            return self

        def cpu(self) -> object:
            return self

        def tolist(self) -> list[list[list[float]]]:
            return [[[0.9, 0.4]] * 20]

    class Model:
        def diarize(self, **_kwargs: object) -> tuple[list[list[str]], list[Probabilities]]:
            return (
                [["0.100 0.900 speaker_0", "1.200 2.000 speaker_1"]],
                [Probabilities()],
            )

    monkeypatch.setattr(identity_diarization, "_extract_audio_wav", lambda _path: b"wav")
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"placeholder")

    segments = await NemoSortformerSpeechDiarizer(Model(), device="cuda").segment_file(
        media,
        duration_ms=2_000,
    )

    assert [(item.start_ms, item.end_ms, item.speaker_label) for item in segments] == [
        (100, 900, "speaker_0"),
        (1_200, 2_000, "speaker_1"),
    ]
    assert [item.confidence for item in segments] == pytest.approx([0.8, 0.4])


def test_speech_fusion_attaches_unambiguous_asr_and_marks_overlap_unsafe() -> None:
    transcripts = (
        SpeechSegment("asr_0", 100, 800, 0.9, "hello"),
        SpeechSegment("asr_ambiguous", 800, 1_200, 0.9, "unclear"),
        SpeechSegment("asr_1", 1_200, 1_800, 0.8, "there"),
    )
    turns = (
        SpeechSegment("turn_0", 0, 1_000, 0.95, speaker_label="speaker_0"),
        SpeechSegment("turn_1", 1_000, 2_000, 0.95, speaker_label="speaker_1"),
    )

    fused = fuse_speech_segments(transcripts, turns)

    assert [(item.transcript, item.speaker_label, item.confidence) for item in fused] == [
        ("hello", "speaker_0", 0.9),
        ("unclear", None, 0.9),
        ("there", "speaker_1", 0.8),
    ]

    overlapping = fuse_speech_segments(
        (),
        (
            SpeechSegment("turn_0", 0, 1_100, 0.95, speaker_label="speaker_0"),
            SpeechSegment("turn_1", 900, 2_000, 0.95, speaker_label="speaker_1"),
        ),
    )
    assert [item.confidence for item in overlapping] == [0.5, 0.5]


async def test_av_identity_segment_runs_the_complete_revocable_handoff(tmp_path: Path) -> None:
    class Faces:
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
                    sample_id="face_0",
                    start_ms=0,
                    end_ms=1_000,
                    confidence=0.95,
                    visual_bbox_xyxy=(0.1, 0.1, 0.4, 0.8),
                    embedding=(1.0, 0.0),
                ),
            )

    class Transcriber:
        async def segment_file(self, _media_path: Path) -> tuple[SpeechSegment, ...]:
            return (SpeechSegment("asr_0", 0, 1_000, 0.9, "Pass the tool."),)

    class Diarizer:
        async def segment_file(
            self,
            _media_path: Path,
            *,
            duration_ms: int,
        ) -> tuple[SpeechSegment, ...]:
            assert duration_ms == 1_000
            return (SpeechSegment("turn_0", 0, 1_000, 0.9, speaker_label="speaker_0"),)

    class Speaker:
        def encode_media_segments(
            self,
            _media_path: Path,
            segments: tuple[SpeechSegment, ...],
        ) -> tuple[tuple[float, ...], ...]:
            return tuple((0.0, 1.0) for _ in segments)

    class Matcher:
        calls = 0
        model_reference = ModelReference("test/asd", "1")

        async def match_file(
            self,
            media_path: Path,
            *,
            audio_path: Path | None = None,
            tenant_id: str,
            observation_id: str,
            face_observations: tuple[IdentityObservationInput, ...],
            voice_observations: tuple[IdentityObservationInput, ...],
        ) -> tuple[FaceVoiceAssociationEvidence, ...]:
            assert media_path == video.resolve()
            assert audio_path == audio.resolve()
            assert voice_observations[0].transcript == "Pass the tool."
            self.calls += 1
            if self.calls > 1:
                return ()
            return (
                FaceVoiceAssociationEvidence(
                    tenant_id=tenant_id,
                    source_observation_id=observation_id,
                    evidence_id="association_0",
                    face_identity_id=face_observations[0].identity_id,
                    voice_identity_id=voice_observations[0].identity_id,
                    start_ms=0,
                    end_ms=1_000,
                    confidence=0.9,
                    model_reference=ModelReference("test/asd", "1"),
                ),
            )

    video = tmp_path / "clip.mp4"
    audio = tmp_path / "clip.wav"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    memory = SQLiteIdentityMemory(
        tmp_path / "identity.sqlite3",
        device_id="robot_01",
        encryption_key=AESGCM.generate_key(bit_length=256),
    )
    faces = Faces()
    diarizer = Diarizer()
    matcher = Matcher()
    thresholds = IdentityMatchingThresholds(
        face_similarity=0.8,
        face_margin=0.05,
        voice_similarity=0.8,
        voice_margin=0.05,
        association_observations=1,
        association_duration_ms=1,
        association_confidence=0.8,
        association_margin=0.05,
    )

    result = await recognize_identities_in_av_segment(
        video,
        audio_path=audio,
        tenant_id="tenant_01",
        observation_id="observation_01",
        duration_ms=1_000,
        memory=memory,
        face_encoder=cast(InsightFaceVideoEncoder, faces),
        speech_transcriber=cast(FunASRSpeechTranscriber, Transcriber()),
        speaker_diarizer=cast(NemoSortformerSpeechDiarizer, diarizer),
        speaker_encoder=cast(ERes2NetV2SpeakerEncoder, Speaker()),
        active_speaker_matcher=cast(OpenAIVisualActiveSpeakerMatcher, matcher),
        thresholds=thresholds,
        parallel_model_inference=False,
    )

    face, voice = result.identity_observations
    assert face.kind is IdentityKind.FACE
    assert voice.kind is IdentityKind.VOICE
    assert voice.identity_id == face.identity_id
    assert voice.transcript == "Pass the tool."
    assert len(result.association_evidence) == 1

    later = await recognize_identities_in_av_segment(
        video,
        audio_path=audio,
        tenant_id="tenant_01",
        observation_id="observation_02",
        duration_ms=1_000,
        memory=memory,
        face_encoder=cast(InsightFaceVideoEncoder, faces),
        speech_transcriber=cast(FunASRSpeechTranscriber, Transcriber()),
        speaker_diarizer=cast(NemoSortformerSpeechDiarizer, diarizer),
        speaker_encoder=cast(ERes2NetV2SpeakerEncoder, Speaker()),
        active_speaker_matcher=cast(OpenAIVisualActiveSpeakerMatcher, matcher),
        thresholds=thresholds,
        parallel_model_inference=False,
    )

    later_face, later_voice = later.identity_observations
    assert later.association_evidence == ()
    assert later_face.identity_id == face.identity_id
    assert later_voice.identity_id == face.identity_id


async def test_visual_active_speaker_returns_revocable_edge_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        messages = cast(list[dict[str, object]], payload["messages"])
        content = cast(list[dict[str, object]], messages[1]["content"])
        assert {part["type"] for part in content} == {"text", "video_url"}
        assert "off-screen person" in cast(str, messages[0]["content"])
        assert "F0" in cast(str, content[0]["text"])
        video = next(part for part in content if part["type"] == "video_url")
        assert cast(dict[str, str], video["video_url"])["url"].endswith("YW5ub3RhdGVk")
        return _streaming_response(
            {"matches": [{"speech_index": 0, "face_identity_id": "face_01", "confidence": 0.9}]}
        )

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"real-media-placeholder")
    monkeypatch.setattr(
        identity_diarization,
        "_annotated_face_video",
        lambda path, _faces, _maximum_bytes, _audio_path=None: (path, b"annotated"),
    )
    matcher = OpenAIVisualActiveSpeakerMatcher(
        _client(respond),
        model_revision="deployment-revision",
    )
    try:
        evidence = await matcher.match_file(
            media,
            tenant_id="tenant_01",
            observation_id="observation_01",
            face_observations=(
                _identity("face_01", IdentityKind.FACE, 100, 900, bbox=(0.1, 0.2, 0.4, 0.8)),
            ),
            voice_observations=(
                _identity("voice_01", IdentityKind.VOICE, 200, 800, transcript="Hello."),
            ),
        )
    finally:
        await matcher.close()

    assert len(evidence) == 1
    assert evidence[0].face_identity_id == "face_01"
    assert evidence[0].voice_identity_id == "voice_01"
    assert (evidence[0].start_ms, evidence[0].end_ms) == (200, 800)
    assert evidence[0].model_reference == ModelReference(
        "qwen3.8-max",
        "deployment-revision",
    )


async def test_visual_active_speaker_skips_observation_scoped_voices(tmp_path: Path) -> None:
    async def fail(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("observation-scoped speech must not be biometrically associated")

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"real-media-placeholder")
    matcher = OpenAIVisualActiveSpeakerMatcher(_client(fail), model_revision="revision")
    try:
        evidence = await matcher.match_file(
            media,
            tenant_id="tenant_01",
            observation_id="observation_01",
            face_observations=(
                _identity("face_01", IdentityKind.FACE, 0, 900, bbox=(0.1, 0.2, 0.4, 0.8)),
            ),
            voice_observations=(
                _identity(
                    "voice_weak",
                    IdentityKind.VOICE,
                    100,
                    700,
                    transcript="嗯。",
                    scope=IdentityScope.OBSERVATION,
                ),
            ),
        )
    finally:
        await matcher.close()

    assert evidence == ()


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg is not installed",
)
def test_face_anchor_annotation_muxes_the_audio_sidecar(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    audio = tmp_path / "clip.wav"
    subprocess.run(
        (
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=black:size=64x64:rate=2:duration=1",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(media),
        ),
        check=True,
        timeout=30,
    )
    subprocess.run(
        (
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=1",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            str(audio),
        ),
        check=True,
        timeout=30,
    )

    _, annotated = identity_diarization._annotated_face_video(
        media,
        (_identity("face_01", IdentityKind.FACE, 100, 900, bbox=(0.1, 0.2, 0.4, 0.8)),),
        1_000_000,
        audio,
    )

    annotated_path = tmp_path / "annotated.mp4"
    annotated_path.write_bytes(annotated)
    probe = subprocess.run(
        (
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type",
            "-of",
            "json",
            str(annotated_path),
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    metadata = json.loads(probe.stdout)
    assert {stream["codec_type"] for stream in metadata["streams"]} == {"video", "audio"}
    assert float(metadata["format"]["duration"]) >= 0.9


def _segmenter(
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> OpenAIAVSpeechSegmenter:
    return OpenAIAVSpeechSegmenter(_client(handler))


def _client(
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="unit-test-key",
        base_url=normalize_openai_base_url("https://vlm.example.test/api/v1/chat/completions"),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )


def _identity(
    identity_id: str,
    kind: IdentityKind,
    start_ms: int,
    end_ms: int,
    *,
    transcript: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    scope: IdentityScope = IdentityScope.DEVICE,
) -> IdentityObservationInput:
    return IdentityObservationInput(
        identity_id=identity_id,
        kind=kind,
        start_ms=start_ms,
        end_ms=end_ms,
        confidence=0.95,
        model_id="test/model",
        model_revision="1",
        scope=scope,
        transcript=transcript,
        visual_bbox_xyxy=bbox,
    )


def _streaming_response(payload: object) -> httpx.Response:
    event = {
        "id": "completion_01",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "qwen3.8-max",
        "system_fingerprint": "provider-fingerprint",
        "choices": [
            {
                "index": 0,
                "delta": {"content": json.dumps(payload)},
                "finish_reason": "stop",
            }
        ],
    }
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n",
    )
