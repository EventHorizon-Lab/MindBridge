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
from typing import Any, cast

import httpx
import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from openai import AsyncOpenAI

from mindbridge.contracts import IdentityObservationInput
from mindbridge.core import (
    IdentityKind,
    IdentityScope,
    ModelOutputError,
    ModelReference,
    ModelUnavailableError,
)
from mindbridge.edge import identity_diarization
from mindbridge.edge.identity import FaceVoiceAssociationEvidence, SQLiteIdentityMemory
from mindbridge.edge.identity_diarization import (
    FUNASR_RECIPES,
    FUNASR_SENSEVOICE_MODEL_ID,
    ActiveSpeakerMatcher,
    FunASRAutoModelPipeline,
    FunASRRecipe,
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
    analysis = await FunASRAutoModelPipeline(Pipeline(), device="cuda").analyze_file(media)

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

    pipeline = FunASRAutoModelPipeline.load(device="cuda")

    assert pipeline.device == "cuda"
    assert arguments["spk_model"] == CAMPPLUS_MODEL.model_id
    # The default is Fun-ASR-Nano on `AutoModel`. Asserted here because a default nothing
    # names is a default nothing notices changing.
    assert arguments["model"] == identity_diarization.FUNASR_NANO_MODEL_ID
    # Off, and asserted rather than merely absent from the recipe: under this flag upstream
    # pip-installs a downloaded requirements.txt into the live venv and imports downloaded
    # code. Fun-ASR-Nano needs none of it -- funasr registers the architecture natively.
    assert "trust_remote_code" not in arguments
    assert arguments["vad_kwargs"] == {"max_single_segment_time": 30_000}
    # Fun-ASR-Nano punctuates its own output, so loading a punctuation model would only cost
    # weights: upstream skips it whenever the result already carries timestamps.
    assert "punc_model" not in arguments
    # Unpinned by default, which is upstream's "master". Safe only because no downloaded code
    # runs; a deployment that has measured a checkpoint should still pin it.
    assert "model_revision" not in arguments


def test_funasr_recipe_can_pin_the_revision_it_trusts_remote_code_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments: dict[str, object] = {}

    def auto_model(**kwargs: object) -> object:
        arguments.update(kwargs)
        return _StubFunASRPipeline([])

    funasr = ModuleType("funasr")
    funasr.AutoModel = auto_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "funasr", funasr)
    monkeypatch.setattr(identity_diarization, "select_torch_device", lambda _device: "cuda")

    FunASRAutoModelPipeline.load(
        device="cuda",
        recipe=FunASRRecipe(
            model_id=identity_diarization.FUNASR_NANO_MODEL_ID,
            vad_max_single_segment_ms=30_000,
            trust_remote_code=True,
            revision="v1.0.0",
        ),
    )

    assert arguments["model_revision"] == "v1.0.0"


async def test_funasr_rejects_unrecoverably_untimed_speech(tmp_path: Path) -> None:
    class Pipeline:
        def generate(self, **_kwargs: object) -> list[dict[str, object]]:
            return [{"text": "speech was detected", "sentence_info": []}]

    media = tmp_path / "clip.wav"
    media.write_bytes(b"placeholder")

    with pytest.raises(ModelOutputError, match="without timed"):
        await FunASRAutoModelPipeline(Pipeline(), device="cuda").analyze_file(media)


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


def test_funasr_accepts_a_per_token_sentence_text() -> None:
    """FunASR emits `text` as a plain string on some decode paths and a per-token list on
    others. Rejecting the list shape discarded the whole clip: measured on a conversational
    corpus, 25 of 40 clips produced no transcript at all for this reason alone."""
    segments = identity_diarization._funasr_segments(
        [{"start": 0, "end": 1_000, "text": ["哦", "好", "的"], "spk": 0}],
        confidence=0.8,
    )

    assert [segment.transcript for segment in segments] == ["哦好的"]


def test_funasr_still_rejects_a_sentence_with_no_usable_text() -> None:
    """The shape widened; the guard did not go away."""
    for broken in ([], [""], ["ok", 7], None, 12):
        with pytest.raises(ModelOutputError):
            identity_diarization._funasr_segments(
                [{"start": 0, "end": 1_000, "text": broken, "spk": 0}],
                confidence=0.8,
            )


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
        speech_pipeline=cast(FunASRAutoModelPipeline, SpeechPipeline()),
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
        speech_pipeline=cast(FunASRAutoModelPipeline, SpeechPipeline()),
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


def test_funasr_recipe_refuses_a_model_without_the_capabilities_mindbridge_needs() -> None:
    """Reject the composition, not the inference: `SpeechAnalysis` cannot be filled without
    VAD (no timed spans) or a speaker model (no centroid to match a voiceprint against), and
    finding that out costs several GiB of weights and a full decode if it is not checked here.
    """
    with pytest.raises(ValueError, match="timed spans"):
        FunASRRecipe(model_id=FUNASR_SENSEVOICE_MODEL_ID, vad_model="")
    with pytest.raises(ValueError, match="voiceprint centroids"):
        FunASRRecipe(model_id=FUNASR_SENSEVOICE_MODEL_ID, speaker_model="")
    with pytest.raises(ValueError, match="model id"):
        FunASRRecipe(model_id="  ")


def test_funasr_recipes_compose_each_model_the_way_upstream_does() -> None:
    paraformer = FUNASR_RECIPES["paraformer"].auto_model_arguments()
    sensevoice = FUNASR_RECIPES["sensevoice"].auto_model_arguments()
    nano = FUNASR_RECIPES["fun-asr-nano"].auto_model_arguments()

    # Only Paraformer predicts the character timestamps punctuation has to align to, so it is
    # the only recipe that pays for a punctuation model.
    assert paraformer["punc_model"] == identity_diarization.FUNASR_PUNCTUATION_MODEL_ID
    assert "vad_kwargs" not in paraformer
    assert "punc_model" not in sensevoice
    assert sensevoice["vad_kwargs"] == {"max_single_segment_time": 30_000}
    assert "punc_model" not in nano
    assert "trust_remote_code" not in nano
    # Every recipe still has to satisfy the contract, whatever else it drops.
    for arguments in (paraformer, sensevoice, nano):
        assert arguments["vad_model"] and arguments["spk_model"] == CAMPPLUS_MODEL.model_id


def test_funasr_swaps_the_model_without_touching_the_call_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments: dict[str, object] = {}

    def auto_model(**kwargs: object) -> object:
        arguments.update(kwargs)
        return _StubFunASRPipeline([])

    funasr = ModuleType("funasr")
    funasr.AutoModel = auto_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "funasr", funasr)
    monkeypatch.setattr(identity_diarization, "select_torch_device", lambda _device: "cuda")

    FunASRAutoModelPipeline.load(device="cuda", recipe="sensevoice")

    assert arguments["model"] == FUNASR_SENSEVOICE_MODEL_ID
    assert arguments["spk_model"] == CAMPPLUS_MODEL.model_id
    assert "punc_model" not in arguments

    with pytest.raises(ValueError, match="sensevoice"):
        FunASRAutoModelPipeline.load(device="cuda", recipe="whisper")


async def test_funasr_strips_model_special_tokens_and_drops_silent_spans(
    tmp_path: Path,
) -> None:
    """SenseVoice tags language, emotion and event inline, and reports tag-only spans for VAD
    windows it heard no speech in. Left alone the tags become claim text, and treating a
    tag-only span as malformed would take every other sentence in the clip down with it.
    """
    pipeline = _StubFunASRPipeline(
        [
            {
                "text": "<|zh|><|NEUTRAL|><|Speech|><|woitn|>今天天气不错",
                "sentence_info": [
                    {
                        "start": 0,
                        "end": 900,
                        "sentence": "<|zh|><|NEUTRAL|><|Speech|><|woitn|>今天天气不错",
                        "spk": 0,
                    },
                    {"start": 1_000, "end": 1_400, "sentence": "<|zh|><|EMO_UNKNOWN|>", "spk": 0},
                ],
                "spk_embedding_center": [[1.0, 0.0]],
            }
        ]
    )
    media = tmp_path / "clip.wav"
    media.write_bytes(b"placeholder")

    analysis = await FunASRAutoModelPipeline(pipeline, device="cuda").analyze_file(media)

    assert [(item.start_ms, item.end_ms, item.transcript) for item in analysis.segments] == [
        (0, 900, "今天天气不错")
    ]


async def test_funasr_treats_a_fully_tagged_clip_as_silence(tmp_path: Path) -> None:
    """Silence is a different shape per model: Paraformer returns "", SenseVoice still tags the
    span. Both mean nobody spoke, so neither may reach the untimed-speech error.
    """
    media = tmp_path / "clip.wav"
    media.write_bytes(b"placeholder")

    analysis = await FunASRAutoModelPipeline(
        _StubFunASRPipeline(
            [{"text": "<|zh|><|NEUTRAL|><|Speech|><|woitn|>", "sentence_info": []}]
        ),
        device="cuda",
    ).analyze_file(media)

    assert analysis == SpeechAnalysis(segments=(), speaker_embeddings=())


async def test_nano_vllm_batches_vad_spans_and_normalizes_to_the_same_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The engine only transcribes. This checks the composition around it: FSMN-VAD picks the
    spans to batch, CTC alignment tightens them inside those spans, CAM++ chunks are handed
    over in seconds and in chunk order, and `postprocess` is asked for the centroids upstream's
    own server throws away.
    """
    calls = _install_nano_stubs(monkeypatch, duration_ms=4_000)
    media = tmp_path / "clip.wav"
    media.write_bytes(b"placeholder")

    vad = _StubFunASRPipeline(
        [
            {
                "value": [
                    [0, 1_000],
                    [1_500, 2_500],
                    # Below the 300 ms floor: too little audio to place a speaker, so upstream
                    # drops it rather than transcribing it.
                    [2_600, 2_700],
                ]
            }
        ]
    )
    engine = _StubNanoEngine(
        [
            # CTC alignment, relative to the span: 0.1s-0.8s inside a span starting at 0 ms.
            {
                "text": "你好",
                "timestamps": [
                    {"token": "你", "start_time": 0.1, "end_time": 0.4},
                    {"token": "好", "start_time": 0.4, "end_time": 0.8},
                ],
            },
            # Forced alignment can fail; the VAD span is then the only timing available.
            {"text": "Hello world."},
        ]
    )
    speaker = _StubFunASRPipeline([{"spk_embedding": _StubTensor()}])

    analysis = await identity_diarization.FunASRNanoVLLMPipeline(
        engine, vad, speaker, device="cuda"
    ).analyze_file(media)

    assert engine.batch_sizes == [2]
    assert engine.keywords["repetition_penalty"] == 1.0
    assert [
        (item.start_ms, item.end_ms, item.transcript, item.speaker_label)
        for item in analysis.segments
    ] == [(100, 800, "你好", "0"), (1_500, 2_500, "Hello world.", "1")]
    assert [item.embedding for item in analysis.speaker_embeddings] == [(1.0, 0.0), (0.0, 1.0)]
    # `sv_chunk` works in seconds, and the chunks reach `postprocess` in the order the labels
    # come back in -- sorting them without reordering the embeddings would mislabel speakers.
    assert calls["sv_chunk_spans"] == [(0.1, 0.8), (1.5, 2.5)]
    assert calls["postprocess_chunk_order"] == [0.1, 1.5]
    assert calls["return_spk_center"] is True


async def test_nano_vllm_reports_silence_rather_than_inventing_a_span(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Upstream's server transcribes the whole clip when VAD finds nothing, because a caller
    who posted a file asked for it. Here an empty VAD result is the answer.
    """
    _install_nano_stubs(monkeypatch, duration_ms=2_000)
    media = tmp_path / "clip.wav"
    media.write_bytes(b"placeholder")
    engine = _StubNanoEngine([])

    analysis = await identity_diarization.FunASRNanoVLLMPipeline(
        engine,
        _StubFunASRPipeline([{"value": []}]),
        _StubFunASRPipeline([]),
        device="cuda",
    ).analyze_file(media)

    assert analysis == SpeechAnalysis(segments=(), speaker_embeddings=())
    assert engine.batch_sizes == []


def test_nano_vllm_ctc_alignment_stays_inside_its_vad_span() -> None:
    """The alignment is measured inside one span, so an offset that would run past the span --
    or collapse it -- has to fall back to the span instead of producing a bogus timeline.
    """
    aligned: list[object] = [{"token": "a", "start_time": 0.2, "end_time": 0.5}]
    assert identity_diarization._nano_span(aligned, 1_000, 2_000) == (1_200, 1_500)
    assert identity_diarization._nano_span(aligned, 1_000, 1_100) == (1_000, 1_100)
    assert identity_diarization._nano_span(
        [cast(object, {"start_time": 0.0, "end_time": 0.0})], 5, 9
    ) == (5, 9)
    assert identity_diarization._nano_span([cast(object, {"token": "a"})], 0, 100) is None


def test_nano_vllm_refuses_to_pretend_it_can_run_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine is installed and importable here on purpose. Without that, an absent vLLM
    would raise a ModelUnavailableError of its own and the test would pass whether or not the
    device is ever checked -- which is what it did until this stub was added.
    """
    loaded: list[dict[str, object]] = []

    class _Engine:
        @staticmethod
        def from_pretrained(**kwargs: object) -> object:
            loaded.append(kwargs)
            return _StubNanoEngine([])

    inference_vllm = ModuleType("funasr.models.fun_asr_nano.inference_vllm")
    inference_vllm.FunASRNanoVLLM = _Engine  # type: ignore[attr-defined]
    funasr = ModuleType("funasr")
    funasr.AutoModel = lambda **_kwargs: _StubFunASRPipeline([])  # type: ignore[attr-defined]
    for name, module in (
        ("funasr", funasr),
        ("funasr.models", ModuleType("funasr.models")),
        ("funasr.models.fun_asr_nano", ModuleType("funasr.models.fun_asr_nano")),
        ("funasr.models.fun_asr_nano.inference_vllm", inference_vllm),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(identity_diarization, "select_torch_device", lambda _device: "cpu")

    with pytest.raises(ModelUnavailableError, match="needs CUDA"):
        identity_diarization.FunASRNanoVLLMPipeline.load(device="cpu")
    assert loaded == []

    monkeypatch.setattr(identity_diarization, "select_torch_device", lambda _device: "cuda")
    pipeline = identity_diarization.FunASRNanoVLLMPipeline.load(device="cuda")

    assert pipeline.device == "cuda"
    assert loaded and loaded[0]["model"] == identity_diarization.FUNASR_NANO_MODEL_ID


class _StubFunASRPipeline:
    def __init__(self, output: list[dict[str, object]]) -> None:
        self._output = output
        self.keywords: dict[str, object] = {}

    def generate(self, **kwargs: object) -> list[dict[str, object]]:
        self.keywords = kwargs
        return self._output


class _StubNanoEngine:
    def __init__(self, output: list[dict[str, object]]) -> None:
        self._output = output
        self.batch_sizes: list[int] = []
        self.keywords: dict[str, object] = {}

    def generate(self, *, inputs: list[object], **kwargs: object) -> list[dict[str, object]]:
        self.batch_sizes.append(len(inputs))
        self.keywords = kwargs
        return self._output


class _StubTensor:
    """Enough of a torch tensor for the CAM++ hand-off, backed by real vectors."""

    def __init__(self, rows: tuple[tuple[float, ...], ...] = ((1.0, 0.0), (0.0, 1.0))) -> None:
        self._rows = rows

    def cpu(self) -> "_StubTensor":
        return self

    def detach(self) -> "_StubTensor":
        return self

    def numpy(self) -> object:
        import numpy

        return numpy.asarray(self._rows)


def _install_nano_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    duration_ms: int,
) -> dict[str, object]:
    """Stand in for the upstream primitives the vLLM path composes, recording how it calls them."""
    import numpy

    calls: dict[str, object] = {}
    waveform = _StubWaveform(numpy.zeros(duration_ms * 16, dtype=numpy.float32))

    def sv_chunk(spans: list[list[Any]], *, fs: int) -> list[list[Any]]:
        assert fs == 16_000
        calls["sv_chunk_spans"] = [(round(span[0], 3), round(span[1], 3)) for span in spans]
        return [[span[0], span[1], span[2]] for span in spans]

    def postprocess(
        chunks: list[list[Any]],
        _vad: object,
        labels: object,
        _embeddings: object,
        return_spk_center: bool = False,
    ) -> tuple[list[list[object]], object]:
        calls["postprocess_chunk_order"] = [round(chunk[0], 3) for chunk in chunks]
        calls["return_spk_center"] = return_spk_center
        return [[chunk[0], chunk[1], index] for index, chunk in enumerate(chunks)], numpy.asarray(
            [[1.0, 0.0], [0.0, 1.0]]
        )

    def distribute_spk(sentences: list[dict[str, object]], timeline: list[list[Any]]) -> None:
        for sentence, entry in zip(sentences, timeline, strict=True):
            sentence["spk"] = entry[2]

    campplus_utils = ModuleType("funasr.models.campplus.utils")
    campplus_utils.sv_chunk = sv_chunk  # type: ignore[attr-defined]
    campplus_utils.postprocess = postprocess  # type: ignore[attr-defined]
    campplus_utils.distribute_spk = distribute_spk  # type: ignore[attr-defined]

    cluster_backend = ModuleType("funasr.models.campplus.cluster_backend")
    cluster_backend.ClusterBackend = _StubClusterBackend  # type: ignore[attr-defined]

    load_utils = ModuleType("funasr.utils.load_utils")
    load_utils.load_audio_text_image_video = (  # type: ignore[attr-defined]
        lambda _path, fs: waveform
    )

    torch = ModuleType("torch")
    torch.cat = lambda tensors, dim: tensors[0]  # type: ignore[attr-defined]

    for name, module in (
        ("funasr", ModuleType("funasr")),
        ("funasr.models", ModuleType("funasr.models")),
        ("funasr.models.campplus", ModuleType("funasr.models.campplus")),
        ("funasr.models.campplus.utils", campplus_utils),
        ("funasr.models.campplus.cluster_backend", cluster_backend),
        ("funasr.utils", ModuleType("funasr.utils")),
        ("funasr.utils.load_utils", load_utils),
        ("torch", torch),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return calls


class _StubClusterBackend:
    def __init__(self, *, merge_thr: float) -> None:
        assert merge_thr == 0.78

    def to(self, _device: str) -> "_StubClusterBackend":
        return self

    def __call__(self, _embeddings: object, *, oracle_num: object) -> list[int]:
        assert oracle_num is None
        return [0, 1]


class _StubWaveform:
    def __init__(self, samples: object) -> None:
        self._samples = samples

    def detach(self) -> "_StubWaveform":
        return self

    def cpu(self) -> "_StubWaveform":
        return self

    def numpy(self) -> object:
        return self._samples


def test_engine_selection_follows_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA with vLLM installed goes to vLLM, every other platform goes to AutoModel, and a
    named engine wins over both. Resolving from the environment is only safe because both
    engines fill the whole contract -- neither trades speaker centroids away for speed.
    """
    _stub_engines(monkeypatch)
    device = {"value": "cuda"}
    vllm_installed = {"value": True}
    monkeypatch.setattr(
        identity_diarization, "select_torch_device", lambda _device: device["value"]
    )
    monkeypatch.setattr(
        identity_diarization,
        "find_spec",
        lambda name: object() if name == "vllm" and vllm_installed["value"] else None,
    )

    def engine_for(**kwargs: object) -> str:
        return type(identity_diarization.load_speech_analyzer(**kwargs)).__name__  # type: ignore[arg-type]

    assert engine_for() == "FunASRNanoVLLMPipeline"
    # A GPU host that never installed vLLM is still a GPU host; it just cannot use that
    # engine, and staying on the portable path beats failing at load.
    vllm_installed["value"] = False
    assert engine_for() == "FunASRAutoModelPipeline"
    # Every non-CUDA platform, vLLM installed or not.
    device["value"] = "cpu"
    vllm_installed["value"] = True
    assert engine_for() == "FunASRAutoModelPipeline"

    # Naming vLLM on a device that has no CUDA fails rather than quietly downgrading: the
    # module's standing rule is that an explicit accelerator request is not a suggestion.
    with pytest.raises(ModelUnavailableError, match="needs CUDA"):
        identity_diarization.load_speech_analyzer(engine="vllm")

    # A named engine is honoured against the environment's preference, in both directions.
    device["value"] = "cuda"
    assert engine_for(engine="vllm") == "FunASRNanoVLLMPipeline"
    assert engine_for(engine="AutoModel") == "FunASRAutoModelPipeline"
    vllm_installed["value"] = False
    assert engine_for(engine="vllm") == "FunASRNanoVLLMPipeline"

    with pytest.raises(ValueError, match="unknown speech engine"):
        identity_diarization.load_speech_analyzer(engine="llama.cpp")


def test_engine_selection_never_substitutes_a_different_model_for_the_one_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The vLLM engine serves Fun-ASR-Nano's weights. Picking it for a Paraformer or SenseVoice
    recipe would quietly transcribe with a different model -- a different language profile --
    than the operator configured, which is the exact silent substitution a recipe exists to
    prevent. So the recipe constrains the automatic choice, and an explicit engine that cannot
    run the recipe is refused rather than honoured halfway.
    """
    loaded: dict[str, object] = {}
    _stub_engines(monkeypatch, loaded=loaded)
    monkeypatch.setattr(identity_diarization, "select_torch_device", lambda _device: "cuda")
    monkeypatch.setattr(identity_diarization, "find_spec", lambda _name: object())

    def engine_for(**kwargs: object) -> str:
        return type(identity_diarization.load_speech_analyzer(**kwargs)).__name__  # type: ignore[arg-type]

    # CUDA, vLLM installed, and the default Nano recipe: the fast path applies.
    assert engine_for() == "FunASRNanoVLLMPipeline"
    assert loaded["model"] == identity_diarization.FUNASR_NANO_MODEL_ID

    # Same host, a different model asked for: AutoModel, because vLLM cannot run it.
    assert engine_for(recipe="paraformer") == "FunASRAutoModelPipeline"
    assert engine_for(recipe="sensevoice") == "FunASRAutoModelPipeline"

    # Naming the engine anyway is an error, not a silent model swap.
    with pytest.raises(ValueError, match="cannot run"):
        identity_diarization.load_speech_analyzer(engine="vllm", recipe="sensevoice")

    # A declared recipe can say these are Nano weights under another name -- a local
    # conversion, say -- and then both the id and the pin have to reach the engine. The id is
    # deliberately not the registry's, so forwarding it is observable rather than shadowed by
    # the loader's own default.
    local = FunASRRecipe(
        model_id="/opt/funasr/fun-asr-nano-2512",
        vad_max_single_segment_ms=30_000,
        trust_remote_code=True,
        revision="v1.0.0",
        vllm_servable=True,
    )
    assert engine_for(recipe=local) == "FunASRNanoVLLMPipeline"
    assert loaded["model"] == "/opt/funasr/fun-asr-nano-2512"
    assert loaded["revision"] == "v1.0.0"

    # And a recipe that never claimed servability is refused however Nano-like it looks.
    undeclared = FunASRRecipe(
        model_id=identity_diarization.FUNASR_NANO_MODEL_ID,
        vad_max_single_segment_ms=30_000,
        trust_remote_code=True,
    )
    assert engine_for(recipe=undeclared) == "FunASRAutoModelPipeline"


def test_nano_vllm_refuses_to_drop_a_revision_pin_it_cannot_honour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upstream's `from_pretrained` forwards a revision to ModelScope and drops it on the
    HuggingFace path. Dropping a pin silently is how an unreviewed checkpoint gets loaded under
    `trust_remote_code`, so the combination is refused instead.
    """
    _stub_engines(monkeypatch)
    monkeypatch.setattr(identity_diarization, "select_torch_device", lambda _device: "cuda")

    with pytest.raises(ValueError, match="ModelScope"):
        identity_diarization.FunASRNanoVLLMPipeline.load(
            device="cuda",
            hub="hf",
            recipe=FunASRRecipe(
                model_id=identity_diarization.FUNASR_NANO_MODEL_ID,
                revision="v1",
                vllm_servable=True,
            ),
        )


def test_nano_vllm_composes_the_recipe_rather_than_its_own_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This engine is chosen by the environment -- CUDA plus an importable vLLM -- so any
    composition it does not read is a composition that applies on one host and not another.
    The VAD ceiling is the sharp one: left off, FSMN-VAD applies its own 60000ms default, so
    the recipe's 30s turn ceiling silently doubles on exactly the hosts that have a GPU.
    """
    auto_models: list[dict[str, object]] = []
    _stub_engines(monkeypatch, auto_models=auto_models)
    monkeypatch.setattr(identity_diarization, "select_torch_device", lambda _device: "cuda")

    identity_diarization.FunASRNanoVLLMPipeline.load(device="cuda")

    shipped = FUNASR_RECIPES["fun-asr-nano"]
    vad, speaker = auto_models
    assert vad["model"] == shipped.vad_model
    assert vad["max_single_segment_time"] == shipped.vad_max_single_segment_ms == 30_000
    assert speaker["model"] == shipped.speaker_model

    # Declared models that are deliberately not the registry's, so forwarding them is
    # observable rather than the loader's old defaults happening to agree with the recipe.
    auto_models.clear()
    identity_diarization.FunASRNanoVLLMPipeline.load(
        device="cuda",
        recipe=FunASRRecipe(
            model_id="/opt/funasr/nano-local",
            vad_model="/opt/funasr/vad-local",
            speaker_model="/opt/funasr/speaker-local",
            vad_max_single_segment_ms=12_000,
            vllm_servable=True,
        ),
    )
    vad, speaker = auto_models
    assert vad["model"] == "/opt/funasr/vad-local"
    assert vad["max_single_segment_time"] == 12_000
    assert speaker["model"] == "/opt/funasr/speaker-local"

    # An unset ceiling means upstream's, and is left off rather than passed as None -- which
    # FSMN-VAD would then compare frame counts against.
    auto_models.clear()
    identity_diarization.FunASRNanoVLLMPipeline.load(
        device="cuda",
        recipe=FunASRRecipe(model_id="/opt/funasr/nano-local", vllm_servable=True),
    )
    assert "max_single_segment_time" not in auto_models[0]

    # And loading the engine directly is refused for a recipe it cannot serve, the same way
    # the selector refuses it -- one guard, not one per entry point.
    with pytest.raises(ValueError, match="cannot run"):
        identity_diarization.FunASRNanoVLLMPipeline.load(device="cuda", recipe="sensevoice")


def test_funasr_keeps_the_clip_when_one_vad_span_held_no_speech() -> None:
    """Every recipe without a punctuation model degrades to upstream's `vad_segment` mode,
    which emits one entry per VAD span carrying the model's raw text under `sentence` and
    applies no empty-text filter of its own. Treating that `""` as a malformed result threw
    away every other sentence in the segment -- and only on the backends that have no
    punctuation model, so the same clip transcribed on CUDA and raised on CPU.
    """
    segments = identity_diarization._funasr_segments(
        [
            {"start": 0, "end": 1_000, "sentence": "First.", "spk": 0},
            {"start": 1_000, "end": 2_000, "sentence": "", "spk": 0},
            {"start": 2_000, "end": 3_000, "sentence": "   ", "spk": 1},
            {"start": 3_000, "end": 4_000, "text": "<|zh|><|NEUTRAL|><|Speech|>", "spk": 1},
            {"start": 4_000, "end": 5_000, "sentence": "Second.", "spk": 1},
        ],
        confidence=0.8,
    )

    assert [segment.transcript for segment in segments] == ["First.", "Second."]


def test_no_shipped_recipe_runs_downloaded_code() -> None:
    """Under `trust_remote_code` upstream pip-installs a downloaded requirements.txt into the
    live venv and imports downloaded `model.py`, resolved against "master" when no revision is
    pinned -- so an upstream edit changes what executes on every device. No shipped model
    needs it: funasr registers each of these architectures natively, Fun-ASR-Nano included
    (`@tables.register("model_classes", "FunASRNano")`, present at 1.3.19, this project's
    declared floor), and that checkpoint's own config.yaml names exactly that class. The field
    stays on the recipe for an operator whose fork genuinely needs it.
    """
    for name, recipe in FUNASR_RECIPES.items():
        assert recipe.trust_remote_code is False, name
        assert "trust_remote_code" not in recipe.auto_model_arguments(), name


async def test_funasr_streaming_strips_model_special_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The streaming checkpoint is a parameter now, so it can be one that tags its output the
    way SenseVoice does. Left in, a live caption reads the tags out as if someone said them.
    """

    class Pipeline:
        def generate(self, **_kwargs: object) -> list[dict[str, object]]:
            return [{"text": "<|zh|><|NEUTRAL|><|Speech|> hello"}]

    monkeypatch.setattr(identity_diarization, "_pcm16_float32", lambda _pcm16: object())
    transcriber = FunASRStreamingTranscriber(Pipeline(), device="cuda")

    result = await transcriber.push_pcm16(bytes(19_200), is_final=True)

    # Tags go, whitespace stays: these are deltas that get concatenated, so the leading space
    # is word spacing and a `.strip()` here would silently join two words.
    assert result.text == " hello"


def _stub_engines(
    monkeypatch: pytest.MonkeyPatch,
    *,
    loaded: dict[str, object] | None = None,
    auto_models: list[dict[str, object]] | None = None,
) -> None:
    """Let the selector build either backend without any real weights."""

    def from_pretrained(**kwargs: object) -> _StubNanoEngine:
        if loaded is not None:
            loaded.clear()
            loaded.update(kwargs)
        return _StubNanoEngine([])

    inference_vllm = ModuleType("funasr.models.fun_asr_nano.inference_vllm")
    inference_vllm.FunASRNanoVLLM = type(  # type: ignore[attr-defined]
        "_Engine",
        (),
        {"from_pretrained": staticmethod(from_pretrained)},
    )

    def auto_model(**kwargs: object) -> _StubFunASRPipeline:
        if auto_models is not None:
            auto_models.append(kwargs)
        return _StubFunASRPipeline([])

    funasr = ModuleType("funasr")
    funasr.AutoModel = auto_model  # type: ignore[attr-defined]
    for name, module in (
        ("funasr", funasr),
        ("funasr.models", ModuleType("funasr.models")),
        ("funasr.models.fun_asr_nano", ModuleType("funasr.models.fun_asr_nano")),
        ("funasr.models.fun_asr_nano.inference_vllm", inference_vllm),
    ):
        monkeypatch.setitem(sys.modules, name, module)
