"""Contract check for OpenAI-SDK audiovisual speaker segmentation."""

import asyncio
import json
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Coroutine
from pathlib import Path
from types import ModuleType
from typing import cast

import httpx
import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from openai import AsyncOpenAI

from mindbridge.contracts import IdentityObservationInput
from mindbridge.core import IdentityKind, IdentityScope, ModelOutputError, ModelReference
from mindbridge.edge import identity_diarization
from mindbridge.edge.identity import FaceVoiceAssociationEvidence, SQLiteIdentityMemory
from mindbridge.edge.identity_diarization import (
    ActiveSpeakerMatcher,
    FunASRSpeechPipeline,
    FunASRStreamingTranscriber,
    IdentityMatchingThresholds,
    SpeechAnalysis,
    SpeechSegmentationPipeline,
    VisualActiveSpeakerPipeline,
    recognize_identities_in_av_segment,
)
from mindbridge.edge.identity_inference import (
    CAMPPLUS_MODEL,
    FaceEmbeddingSample,
    InsightFaceVideoEncoder,
    SpeakerEmbeddingSample,
    SpeechSegment,
)
from mindbridge.models.openai import OpenAIGenerator, normalize_base_url


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


async def test_funasr_returns_integrated_timed_speech_and_speaker_centroids(
    tmp_path: Path,
) -> None:
    class Pipeline:
        def generate(self, **kwargs: object) -> list[dict[str, object]]:
            assert kwargs["return_spk_center"] is True
            assert kwargs["return_spk_res"] is True
            assert kwargs["sentence_timestamp"] is True
            return [
                {
                    "text": "你好。Hello world.",
                    "sentence_info": [
                        {"start": 0, "end": 700, "text": "你好。", "spk": 0},
                        {"start": 1_300, "end": 2_200, "sentence": "Hello world.", "spk": 1},
                    ],
                    "spk_embedding_center": [[1.0, 0.0], [0.0, 1.0]],
                }
            ]

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"placeholder")
    analysis = await FunASRSpeechPipeline(Pipeline(), device="cuda").analyze_file(media)

    assert [
        (item.start_ms, item.end_ms, item.transcript, item.speaker_label)
        for item in analysis.segments
    ] == [
        (0, 700, "你好。", "0"),
        (1_300, 2_200, "Hello world.", "1"),
    ]
    assert [item.speaker_label for item in analysis.speaker_embeddings] == ["0", "1"]
    assert [item.embedding for item in analysis.speaker_embeddings] == [
        (1.0, 0.0),
        (0.0, 1.0),
    ]


def test_funasr_loads_registered_integrated_speaker_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments: dict[str, object] = {}

    class Pipeline:
        def generate(self, **_kwargs: object) -> list[dict[str, object]]:
            return []

    def auto_model(**kwargs: object) -> Pipeline:
        arguments.update(kwargs)
        return Pipeline()

    funasr = ModuleType("funasr")
    funasr.AutoModel = auto_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "funasr", funasr)
    monkeypatch.setattr(identity_diarization, "select_torch_device", lambda _device: "cuda")

    pipeline = FunASRSpeechPipeline.load(device="cuda")

    assert pipeline.device == "cuda"
    assert arguments["spk_model"] == CAMPPLUS_MODEL.model_id


async def test_funasr_rejects_unrecoverably_untimed_speech(tmp_path: Path) -> None:
    class Pipeline:
        def generate(self, **_kwargs: object) -> list[dict[str, object]]:
            return [{"text": "speech was detected", "sentence_info": []}]

    media = tmp_path / "clip.wav"
    media.write_bytes(b"placeholder")

    with pytest.raises(ModelOutputError, match="without timed"):
        await FunASRSpeechPipeline(Pipeline(), device="cuda").analyze_file(media)


async def test_funasr_streaming_accepts_arbitrary_pcm_chunk_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Pipeline:
        def __init__(self) -> None:
            self.calls: list[bool] = []
            self.cache: object | None = None

        def generate(self, **kwargs: object) -> list[dict[str, object]]:
            self.calls.append(cast(bool, kwargs["is_final"]))
            assert kwargs["chunk_size"] == (0, 10, 5)
            self.cache = kwargs["cache"] if self.cache is None else self.cache
            assert kwargs["cache"] is self.cache
            return [{"text": "你好。"}]

    pipeline = Pipeline()
    monkeypatch.setattr(identity_diarization, "_pcm16_float32", lambda _pcm16: object())
    transcriber = FunASRStreamingTranscriber(pipeline, device="cuda")

    partial = await transcriber.push_pcm16(bytes(9_600))
    final = await transcriber.push_pcm16(bytes(9_600), is_final=True)

    assert partial.text == ""
    assert partial.audio_end_ms == 300
    assert final == identity_diarization.StreamingTranscript("你好。", 600, True)
    assert pipeline.calls == [True]
    with pytest.raises(RuntimeError, match="already final"):
        await transcriber.push_pcm16(b"")


async def test_funasr_streaming_closes_after_model_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Pipeline:
        def generate(self, **_kwargs: object) -> list[dict[str, object]]:
            raise RuntimeError("device failure")

    monkeypatch.setattr(identity_diarization, "_pcm16_float32", lambda _pcm16: object())
    transcriber = FunASRStreamingTranscriber(Pipeline(), device="cuda")

    with pytest.raises(RuntimeError, match="device failure"):
        await transcriber.push_pcm16(bytes(19_200))
    with pytest.raises(RuntimeError, match="already final"):
        await transcriber.push_pcm16(bytes(2))


async def test_funasr_streaming_closes_when_inference_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class Pipeline:
        def generate(self, **_kwargs: object) -> list[dict[str, object]]:
            started.set()
            release.wait(timeout=2)
            return [{"text": "late result"}]

    monkeypatch.setattr(identity_diarization, "_pcm16_float32", lambda _pcm16: object())
    transcriber = FunASRStreamingTranscriber(Pipeline(), device="cuda")
    task = asyncio.create_task(transcriber.push_pcm16(bytes(19_200)))
    assert await asyncio.to_thread(started.wait, 1)

    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(RuntimeError, match="already final"):
            await transcriber.push_pcm16(bytes(2))
    finally:
        release.set()


def test_funasr_marks_overlapping_speakers_unsafe_for_enrollment() -> None:
    segments = identity_diarization._funasr_segments(
        [
            {"start": 0, "end": 1_000, "text": "First.", "spk": 0},
            {"start": 800, "end": 1_500, "text": "Second.", "spk": 1},
        ],
        confidence=0.8,
    )

    assert [segment.confidence for segment in segments] == [0.5, 0.5]


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

    class SpeechPipeline:
        async def analyze_file(self, _media_path: Path) -> SpeechAnalysis:
            return SpeechAnalysis(
                segments=(
                    SpeechSegment(
                        "turn_0",
                        0,
                        1_000,
                        0.9,
                        "Pass the tool.",
                        speaker_label="0",
                    ),
                ),
                speaker_embeddings=(SpeakerEmbeddingSample("0", (0.0, 1.0)),),
            )

    class Matcher:
        calls = 0
        model_reference = ModelReference("test/asd")

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
                    model_reference=ModelReference("test/asd"),
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
        speech_pipeline=cast(FunASRSpeechPipeline, SpeechPipeline()),
        active_speaker_matcher=cast(ActiveSpeakerMatcher, matcher),
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
        speech_pipeline=cast(FunASRSpeechPipeline, SpeechPipeline()),
        active_speaker_matcher=cast(ActiveSpeakerMatcher, matcher),
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
    matcher = _VisualHarness(_generator(respond))
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
    assert evidence[0].model_reference == ModelReference("qwen3.8-max")


async def test_visual_active_speaker_skips_observation_scoped_voices(tmp_path: Path) -> None:
    async def fail(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("observation-scoped speech must not be biometrically associated")

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"real-media-placeholder")
    matcher = _VisualHarness(_generator(fail))
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
) -> "_SpeechHarness":
    return _SpeechHarness(_generator(handler))


def _generator(
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> OpenAIGenerator:
    return OpenAIGenerator(
        AsyncOpenAI(
            api_key="unit-test-key",
            base_url=normalize_base_url("https://vlm.example.test/api/v1/chat/completions"),
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            max_retries=0,
        ),
        ModelReference("qwen3.8-max"),
    )


class _SpeechHarness(SpeechSegmentationPipeline):
    def __init__(self, generator: OpenAIGenerator) -> None:
        super().__init__(generator)
        self._owned_generator = generator

    async def close(self) -> None:
        await self._owned_generator.close()


class _VisualHarness(VisualActiveSpeakerPipeline):
    def __init__(self, generator: OpenAIGenerator) -> None:
        super().__init__(
            generator,
            model_reference=ModelReference("qwen3.8-max"),
        )
        self._owned_generator = generator

    async def close(self) -> None:
        await self._owned_generator.close()


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
