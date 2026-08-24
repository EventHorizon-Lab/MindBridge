"""Omni speaker-turn segmentation for the optional edge identity pipeline."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import subprocess
import tempfile
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Protocol, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from mindbridge.application.capabilities import (
    GenerateRequest,
    Generator,
    MediaPart,
    ModelInput,
    TextPart,
)
from mindbridge.application.pipelines.structured import (
    generate_json,
    output_schema,
    unwrap_json_code_fence,
)
from mindbridge.contracts import IdentityObservationInput
from mindbridge.core import (
    IdentityKind,
    IdentityScope,
    MediaKind,
    ModelOutputError,
    ModelReference,
    ModelUnavailableError,
    derive_stable_id,
)
from mindbridge.edge.identity import (
    FaceVoiceAssociationEvidence,
    LocalIdentityMatch,
    SQLiteIdentityMemory,
)
from mindbridge.edge.identity_inference import (
    CAMPPLUS_MODEL,
    InsightFaceVideoEncoder,
    SpeakerEmbeddingSample,
    SpeechSegment,
    recognize_faces_in_video,
    recognize_speakers,
)
from mindbridge.models.compute import has_free_cuda_memory, select_torch_device
from mindbridge.prompts import (
    ACTIVE_SPEAKER_PROMPT,
    MAX_SPEECH_SEGMENTS,
    SEGMENT_SPEECH_PROMPT,
)

FUNASR_ASR_MODEL_ID = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
FUNASR_STREAMING_ASR_MODEL_ID = (
    "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
)
FUNASR_VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
FUNASR_PUNCTUATION_MODEL_ID = "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
_PARALLEL_MODEL_MINIMUM_FREE_CUDA_BYTES = 8 * 1024 * 1024 * 1024

_Transcript = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)
]


class _SegmentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start_ms: Annotated[int, Field(ge=0)]
    end_ms: Annotated[int, Field(ge=0)]
    transcript: _Transcript

    @model_validator(mode="after")
    def require_positive_duration(self) -> _SegmentOutput:
        if self.end_ms <= self.start_ms:
            raise ValueError("speech segment must have positive duration")
        return self


class _DiarizationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    segments: Annotated[tuple[_SegmentOutput, ...], Field(max_length=MAX_SPEECH_SEGMENTS)]

    @model_validator(mode="after")
    def require_chronological_segments(self) -> _DiarizationOutput:
        if any(
            current.start_ms < previous.start_ms
            for previous, current in zip(self.segments, self.segments[1:], strict=False)
        ):
            raise ValueError("speech segments must be chronological")
        return self


class _ActiveSpeakerMatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    speech_index: Annotated[int, Field(ge=0)]
    face_identity_id: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)
    ]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


class _ActiveSpeakerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    matches: Annotated[tuple[_ActiveSpeakerMatch, ...], Field(max_length=MAX_SPEECH_SEGMENTS)]

    @model_validator(mode="after")
    def require_unique_speech_indices(self) -> _ActiveSpeakerOutput:
        if len({match.speech_index for match in self.matches}) != len(self.matches):
            raise ValueError("active-speaker speech indices must be unique")
        return self


@dataclass(frozen=True, slots=True)
class IdentityMatchingThresholds:
    """Device-calibrated biometric and association gates for one AV segment."""

    face_similarity: float
    face_margin: float
    voice_similarity: float
    voice_margin: float
    association_observations: int
    association_duration_ms: int
    association_confidence: float
    association_margin: float

    def __post_init__(self) -> None:
        unit_values = (
            self.face_similarity,
            self.face_margin,
            self.voice_similarity,
            self.voice_margin,
            self.association_confidence,
            self.association_margin,
        )
        if (
            any(not 0.0 <= value <= 1.0 for value in unit_values)
            or self.association_observations <= 0
            or self.association_duration_ms <= 0
        ):
            raise ValueError("identity matching thresholds are invalid")


@dataclass(frozen=True, slots=True)
class IdentitySegmentResult:
    """Cloud-safe identities plus local association evidence from one AV segment."""

    identity_observations: tuple[IdentityObservationInput, ...]
    association_evidence: tuple[FaceVoiceAssociationEvidence, ...]


class ActiveSpeakerMatcher(Protocol):
    """Associate timed speech with visible faces without exposing a provider."""

    @property
    def model_reference(self) -> ModelReference: ...

    async def match_file(
        self,
        media_path: Path,
        *,
        audio_path: Path | None = None,
        tenant_id: str,
        observation_id: str,
        face_observations: tuple[IdentityObservationInput, ...],
        voice_observations: tuple[IdentityObservationInput, ...],
    ) -> tuple[FaceVoiceAssociationEvidence, ...]: ...


class _FunASRPipeline(Protocol):
    def generate(self, **kwargs: object) -> list[dict[str, object]]: ...


@dataclass(frozen=True, slots=True)
class SpeechAnalysis:
    """One integrated FunASR result with timed speech and local speaker centroids."""

    segments: tuple[SpeechSegment, ...]
    speaker_embeddings: tuple[SpeakerEmbeddingSample, ...]


_ACTIVE_SPEAKER_SCHEMA = output_schema("active_speaker_match", _ActiveSpeakerOutput)

_SPEECH_SEGMENT_SCHEMA = output_schema("speech_segmentation", _DiarizationOutput)


class FunASRSpeechPipeline:
    """Run upstream VAD, ASR, punctuation, diarization, and speaker embedding once."""

    def __init__(
        self,
        pipeline: _FunASRPipeline,
        *,
        device: str,
        diarization_confidence: float = 0.8,
    ) -> None:
        if not 0.0 <= diarization_confidence <= 1.0:
            raise ValueError("diarization confidence must be between zero and one")
        self._pipeline = pipeline
        self.device = device
        self._diarization_confidence = diarization_confidence

    @classmethod
    def load(cls, *, device: str | None = None) -> FunASRSpeechPipeline:
        """Load the official integrated FunASR pipeline on CUDA when available."""
        try:
            funasr = __import__("funasr")
        except ImportError as error:
            raise ModelUnavailableError("install FunASR for local speech transcription") from error
        selected_device = select_torch_device(device)
        pipeline = funasr.AutoModel(
            model=FUNASR_ASR_MODEL_ID,
            vad_model=FUNASR_VAD_MODEL_ID,
            punc_model=FUNASR_PUNCTUATION_MODEL_ID,
            spk_model=CAMPPLUS_MODEL.model_id,
            device=selected_device,
            disable_update=True,
            disable_pbar=True,
        )
        return cls(cast(_FunASRPipeline, pipeline), device=selected_device)

    async def analyze_file(self, media_path: Path) -> SpeechAnalysis:
        """Keep the edge loop responsive while local GPU inference runs."""
        return await asyncio.to_thread(self._analyze_file, media_path)

    def _analyze_file(self, media_path: Path) -> SpeechAnalysis:
        media_path = media_path.resolve(strict=True)
        output = self._pipeline.generate(
            input=str(media_path),
            batch_size_s=300,
            return_raw_text=True,
            sentence_timestamp=True,
            return_spk_res=True,
            return_spk_center=True,
            disable_pbar=True,
        )
        if len(output) != 1:
            raise ModelOutputError("FunASR returned an invalid transcription batch")
        result = output[0]
        text = result.get("text")
        if not isinstance(text, str):
            raise ModelOutputError("FunASR returned invalid transcription text")
        if not text.strip():
            return SpeechAnalysis(segments=(), speaker_embeddings=())
        segments = _funasr_segments(
            result.get("sentence_info"),
            confidence=self._diarization_confidence,
        )
        if not segments:
            raise ModelOutputError("FunASR returned text without timed speaker sentences")
        return SpeechAnalysis(
            segments=segments,
            speaker_embeddings=_funasr_speaker_embeddings(result.get("spk_embedding_center")),
        )


@dataclass(frozen=True, slots=True)
class StreamingTranscript:
    """One provisional streaming ASR delta at a monotonic audio offset."""

    text: str
    audio_end_ms: int
    is_final: bool


class FunASRStreamingTranscriber:
    """Feed native 16 kHz mono PCM16 into FunASR's cached streaming checkpoint."""

    def __init__(
        self,
        pipeline: _FunASRPipeline,
        *,
        device: str,
        chunk_size: tuple[int, int, int] = (0, 10, 5),
        encoder_chunk_look_back: int = 4,
        decoder_chunk_look_back: int = 1,
    ) -> None:
        if (
            len(chunk_size) != 3
            or chunk_size[1] <= 0
            or min(encoder_chunk_look_back, decoder_chunk_look_back) < 0
        ):
            raise ValueError("FunASR streaming window is invalid")
        self._pipeline = pipeline
        self.device = device
        self._chunk_size = chunk_size
        self._encoder_chunk_look_back = encoder_chunk_look_back
        self._decoder_chunk_look_back = decoder_chunk_look_back
        self._chunk_bytes = chunk_size[1] * 960 * 2
        self._buffer = bytearray()
        self._cache: dict[str, object] = {}
        self._received_sample_count = 0
        self._closed = False
        self._lock = asyncio.Lock()

    @classmethod
    def load(cls, *, device: str | None = None) -> FunASRStreamingTranscriber:
        """Load the causal Paraformer checkpoint only when live ASR is needed."""
        try:
            funasr = __import__("funasr")
        except ImportError as error:
            raise ModelUnavailableError("install FunASR for native streaming ASR") from error
        selected_device = select_torch_device(device)
        pipeline = funasr.AutoModel(
            model=FUNASR_STREAMING_ASR_MODEL_ID,
            device=selected_device,
            disable_update=True,
            disable_pbar=True,
        )
        return cls(cast(_FunASRPipeline, pipeline), device=selected_device)

    async def push_pcm16(self, pcm16: bytes, *, is_final: bool = False) -> StreamingTranscript:
        """Consume arbitrary chunk boundaries with serialized cache access and backpressure."""
        if len(pcm16) % 2:
            raise ValueError("PCM16 chunks must contain complete little-endian samples")
        async with self._lock:
            if self._closed:
                raise RuntimeError("streaming transcription session is already final")
            if is_final and not pcm16 and self._received_sample_count:
                raise ValueError("set is_final on the last non-empty PCM chunk")
            self._received_sample_count += len(pcm16) // 2
            self._buffer.extend(pcm16)
            texts = []
            try:
                while len(self._buffer) >= self._chunk_bytes or (is_final and self._buffer):
                    chunk_length = min(len(self._buffer), self._chunk_bytes)
                    chunk = bytes(self._buffer[:chunk_length])
                    final_chunk = is_final and chunk_length == len(self._buffer)
                    texts.append(
                        await asyncio.to_thread(self._transcribe_chunk, chunk, final_chunk)
                    )
                    del self._buffer[:chunk_length]
            except (Exception, asyncio.CancelledError):
                self._closed = True
                raise
            self._closed = is_final
            return StreamingTranscript(
                text="".join(texts),
                audio_end_ms=round(self._received_sample_count / 16_000 * 1_000),
                is_final=is_final,
            )

    def _transcribe_chunk(self, pcm16: bytes, is_final: bool) -> str:
        output = self._pipeline.generate(
            input=_pcm16_float32(pcm16),
            cache=self._cache,
            is_final=is_final,
            chunk_size=self._chunk_size,
            encoder_chunk_look_back=self._encoder_chunk_look_back,
            decoder_chunk_look_back=self._decoder_chunk_look_back,
            disable_pbar=True,
        )
        if len(output) != 1 or not isinstance(output[0].get("text"), str):
            raise ModelOutputError("FunASR returned an invalid streaming transcription")
        return cast(str, output[0]["text"])


class VisualActiveSpeakerPipeline:
    """Use an audiovisual VLM only as revocable face↔voice association evidence."""

    def __init__(
        self,
        generator: Generator,
        *,
        model_reference: ModelReference,
        max_output_tokens: int = 2_048,
        maximum_media_bytes: int = 19 * 1024 * 1024,
    ) -> None:
        if min(max_output_tokens, maximum_media_bytes) <= 0:
            raise ValueError("active-speaker model limits must be positive")
        self._generator = generator
        self._model_reference = model_reference
        self._max_output_tokens = max_output_tokens
        self._maximum_media_bytes = maximum_media_bytes

    @property
    def model_reference(self) -> ModelReference:
        """Return the stable deployment identity used to accumulate local evidence."""
        return self._model_reference

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
        """Return evidence only; the encrypted identity memory owns promotion gates."""
        faces = tuple(
            item
            for item in face_observations
            if item.kind is IdentityKind.FACE and item.visual_bbox_xyxy is not None
        )
        voices = tuple(
            item
            for item in voice_observations
            if item.kind is IdentityKind.VOICE and item.scope is IdentityScope.DEVICE
        )
        if not tenant_id.strip() or not observation_id.strip() or not faces or not voices:
            return ()
        _, media_bytes = await asyncio.to_thread(
            _annotated_face_video,
            media_path,
            faces,
            self._maximum_media_bytes,
            audio_path,
        )
        face_label_by_id = {
            identity_id: f"F{index}"
            for index, identity_id in enumerate(dict.fromkeys(item.identity_id for item in faces))
        }
        context = {
            "faces": [
                {
                    "visual_label": face_label_by_id[item.identity_id],
                    **item.model_dump(mode="json", exclude_none=True),
                }
                for item in faces
            ],
            "speech": [
                {
                    "speech_index": index,
                    **item.model_dump(mode="json", exclude_none=True),
                }
                for index, item in enumerate(voices)
            ],
        }
        output, _ = await generate_json(
            self._generator,
            GenerateRequest(
                system_prompt=ACTIVE_SPEAKER_PROMPT.text,
                input=ModelInput(
                    (
                        TextPart(json.dumps(context, ensure_ascii=False)),
                        MediaPart(
                            kind=MediaKind.VIDEO,
                            url=(
                                "data:video/mp4;base64,"
                                f"{base64.b64encode(media_bytes).decode('ascii')}"
                            ),
                            source_uri=str(media_path),
                            frames_per_second=5.0,
                            max_pixels=200_704,
                        ),
                        TextPart("Return the visible-speaker matches JSON now."),
                    )
                ),
                max_output_tokens=self._max_output_tokens,
                output_schema=_ACTIVE_SPEAKER_SCHEMA,
            ),
            _parse_active_speakers,
        )
        face_ids = {item.identity_id for item in faces}
        evidence = []
        for match in output.matches:
            if match.speech_index >= len(voices) or match.face_identity_id not in face_ids:
                raise ModelOutputError("active-speaker output references unknown edge metadata")
            voice = voices[match.speech_index]
            overlapping_faces = tuple(
                face
                for face in faces
                if face.identity_id == match.face_identity_id
                and face.end_ms >= voice.start_ms
                and voice.end_ms >= face.start_ms
            )
            if not overlapping_faces:
                raise ModelOutputError("active-speaker match has no visible temporal overlap")
            start_ms = max(voice.start_ms, min(face.start_ms for face in overlapping_faces))
            end_ms = min(voice.end_ms, max(face.end_ms for face in overlapping_faces))
            if end_ms <= start_ms:
                continue
            evidence.append(
                FaceVoiceAssociationEvidence(
                    tenant_id=tenant_id,
                    source_observation_id=observation_id,
                    evidence_id=derive_stable_id(
                        "active_speaker",
                        observation_id,
                        match.speech_index,
                        match.face_identity_id,
                        voice.identity_id,
                        ACTIVE_SPEAKER_PROMPT.version,
                    ),
                    face_identity_id=match.face_identity_id,
                    voice_identity_id=voice.identity_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    confidence=min(
                        match.confidence,
                        voice.confidence,
                        max(face.confidence for face in overlapping_faces),
                    ),
                    model_reference=self._model_reference,
                )
            )
        return tuple(evidence)


async def recognize_identities_in_av_segment(
    video_path: Path,
    *,
    audio_path: Path | None = None,
    tenant_id: str,
    observation_id: str,
    duration_ms: int,
    memory: SQLiteIdentityMemory,
    face_encoder: InsightFaceVideoEncoder,
    speech_pipeline: FunASRSpeechPipeline,
    thresholds: IdentityMatchingThresholds,
    active_speaker_matcher: ActiveSpeakerMatcher | None = None,
    face_samples_per_second: float | None = None,
    parallel_model_inference: bool | None = None,
) -> IdentitySegmentResult:
    """Run the synchronized AV identity path without sending biometric vectors off-device."""
    if not tenant_id.strip() or not observation_id.strip() or duration_ms <= 0:
        raise ValueError("identity segment metadata is invalid")
    video = await asyncio.to_thread(video_path.resolve, strict=True)
    audio = (
        await asyncio.to_thread(audio_path.resolve, strict=True)
        if audio_path is not None
        else video
    )

    async def recognize_faces() -> tuple[IdentityObservationInput, ...]:
        return await asyncio.to_thread(
            recognize_faces_in_video,
            face_encoder,
            memory,
            video,
            tenant_id=tenant_id,
            observation_id=observation_id,
            minimum_similarity=thresholds.face_similarity,
            minimum_margin=thresholds.face_margin,
            samples_per_second=face_samples_per_second,
        )

    parallel = (
        await asyncio.to_thread(
            has_free_cuda_memory,
            _PARALLEL_MODEL_MINIMUM_FREE_CUDA_BYTES,
        )
        if parallel_model_inference is None
        else parallel_model_inference
    )
    if parallel:
        faces, speech = await asyncio.gather(
            recognize_faces(),
            speech_pipeline.analyze_file(audio),
        )
    else:
        faces = await recognize_faces()
        speech = await speech_pipeline.analyze_file(audio)
    if any(segment.end_ms > duration_ms for segment in speech.segments):
        raise ModelOutputError("FunASR speech segment exceeds the clip duration")
    voices = await asyncio.to_thread(
        recognize_speakers,
        memory,
        speech.segments,
        speech.speaker_embeddings,
        tenant_id=tenant_id,
        observation_id=observation_id,
        minimum_similarity=thresholds.voice_similarity,
        minimum_margin=thresholds.voice_margin,
    )
    evidence = (
        await active_speaker_matcher.match_file(
            video,
            audio_path=audio if audio != video else None,
            tenant_id=tenant_id,
            observation_id=observation_id,
            face_observations=faces,
            voice_observations=voices,
        )
        if active_speaker_matcher is not None
        else ()
    )
    reference = (
        active_speaker_matcher.model_reference if active_speaker_matcher is not None else None
    )
    resolved_voices = await asyncio.to_thread(
        _record_and_resolve_voice_identities,
        memory,
        tenant_id,
        voices,
        evidence,
        reference,
        thresholds,
    )
    observations = tuple(
        sorted(
            (*faces, *resolved_voices),
            key=lambda item: (item.start_ms, item.end_ms, item.kind.value, item.identity_id),
        )
    )
    return IdentitySegmentResult(
        identity_observations=observations,
        association_evidence=evidence,
    )


class SpeechSegmentationPipeline:
    """Turn a Generator into anonymous audiovisual speech turns."""

    def __init__(
        self,
        generator: Generator,
        *,
        max_output_tokens: int = 4_096,
        maximum_media_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if max_output_tokens <= 0:
            raise ValueError("speech segmenter model settings are invalid")
        if maximum_media_bytes <= 0:
            raise ValueError("maximum_media_bytes must be positive")
        self._generator = generator
        self._max_output_tokens = max_output_tokens
        self._maximum_media_bytes = maximum_media_bytes

    async def segment_file(
        self,
        media_path: Path,
        *,
        duration_ms: int,
        confidence: float = 0.8,
    ) -> tuple[SpeechSegment, ...]:
        """Return anonymous turns; voice embeddings and IDs remain a separate local step."""
        if duration_ms <= 0 or not 0.0 <= confidence <= 1.0:
            raise ValueError("clip duration and segment confidence are invalid")
        path, media_bytes = await asyncio.to_thread(
            _read_bounded_media,
            media_path,
            self._maximum_media_bytes,
        )
        media_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
        media_url = f"data:{media_type};base64,{base64.b64encode(media_bytes).decode('ascii')}"
        output, _ = await generate_json(
            self._generator,
            GenerateRequest(
                system_prompt=SEGMENT_SPEECH_PROMPT.text,
                input=ModelInput(
                    (
                        TextPart(json.dumps({"duration_ms": duration_ms}, separators=(",", ":"))),
                        MediaPart(
                            kind=MediaKind.VIDEO,
                            url=media_url,
                            source_uri=str(path),
                            frames_per_second=1.0,
                            max_pixels=200_704,
                        ),
                        TextPart("Return the required speaker-turn JSON now."),
                    )
                ),
                max_output_tokens=self._max_output_tokens,
                output_schema=_SPEECH_SEGMENT_SCHEMA,
            ),
            lambda content: _parse_output(content, duration_ms),
        )
        return tuple(
            SpeechSegment(
                sample_id=f"voice-{segment.start_ms:012d}-{segment.end_ms:012d}-{index:03d}",
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                confidence=confidence,
                transcript=segment.transcript,
            )
            for index, segment in enumerate(output.segments)
        )


def _record_and_resolve_voice_identities(
    memory: SQLiteIdentityMemory,
    tenant_id: str,
    voices: tuple[IdentityObservationInput, ...],
    evidence: tuple[FaceVoiceAssociationEvidence, ...],
    association_model_reference: ModelReference | None,
    thresholds: IdentityMatchingThresholds,
) -> tuple[IdentityObservationInput, ...]:
    for item in evidence:
        memory.record_face_voice_evidence(item)
    if association_model_reference is None:
        return voices
    resolved = []
    for voice in voices:
        if voice.scope is IdentityScope.OBSERVATION:
            resolved.append(voice)
            continue
        match = memory.resolve_identity(
            tenant_id,
            LocalIdentityMatch(
                identity_id=voice.identity_id,
                kind=voice.kind,
                confidence=voice.confidence,
                model_reference=ModelReference(model_id=voice.model_id),
                enrolled_new=False,
            ),
            association_model_reference=association_model_reference,
            minimum_observations=thresholds.association_observations,
            minimum_duration_ms=thresholds.association_duration_ms,
            minimum_confidence=thresholds.association_confidence,
            minimum_margin=thresholds.association_margin,
        )
        resolved.append(voice.model_copy(update={"identity_id": match.identity_id}))
    return tuple(resolved)


def _funasr_segments(value: object, *, confidence: float) -> tuple[SpeechSegment, ...]:
    if not isinstance(value, list) or len(value) > MAX_SPEECH_SEGMENTS:
        raise ModelOutputError("FunASR returned invalid sentence-level diarization")
    segments = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ModelOutputError("FunASR returned an invalid sentence")
        start, end = item.get("start"), item.get("end")
        transcript = item.get("text") or item.get("sentence")
        speaker = item.get("spk")
        if (
            not isinstance(start, (int, float))
            or isinstance(start, bool)
            or not isinstance(end, (int, float))
            or isinstance(end, bool)
            or not isinstance(transcript, str)
            or not transcript.strip()
            or (speaker is not None and not isinstance(speaker, (str, int)))
        ):
            raise ModelOutputError("FunASR returned an invalid sentence")
        start_ms, end_ms = round(start), round(end)
        try:
            segments.append(
                SpeechSegment(
                    sample_id=f"funasr-{start_ms:012d}-{end_ms:012d}-{index:03d}",
                    start_ms=start_ms,
                    end_ms=end_ms,
                    confidence=confidence,
                    transcript=transcript.strip(),
                    speaker_label=str(speaker).strip() if speaker is not None else None,
                )
            )
        except ValueError as error:
            raise ModelOutputError("FunASR returned an invalid sentence range") from error
    if any(current.start_ms < previous.start_ms for previous, current in pairwise(segments)):
        raise ModelOutputError("FunASR sentences must be chronological")
    return tuple(
        replace(segment, confidence=min(segment.confidence, 0.5))
        if any(
            segment.speaker_label != other.speaker_label
            and segment.start_ms < other.end_ms
            and other.start_ms < segment.end_ms
            for other in segments
        )
        else segment
        for segment in segments
    )


def _funasr_speaker_embeddings(value: object) -> tuple[SpeakerEmbeddingSample, ...]:
    if value is None:
        return ()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if not isinstance(value, list):
        raise ModelOutputError("FunASR returned invalid speaker centroids")
    rows = [value] if value and all(isinstance(item, (int, float)) for item in value) else value
    if any(not isinstance(row, list) for row in rows):
        raise ModelOutputError("FunASR returned invalid speaker centroids")
    try:
        return tuple(
            SpeakerEmbeddingSample(
                speaker_label=str(index),
                embedding=tuple(float(cast(int | float, number)) for number in row),
            )
            for index, row in enumerate(rows)
        )
    except (TypeError, ValueError) as error:
        raise ModelOutputError("FunASR returned invalid speaker centroids") from error


def _pcm16_float32(pcm16: bytes) -> object:
    try:
        numpy = __import__("numpy")
    except ImportError as error:
        raise ModelUnavailableError("FunASR streaming requires NumPy") from error
    return numpy.frombuffer(pcm16, dtype="<i2").astype("float32") / 32768.0


def _parse_output(content: str, duration_ms: int) -> _DiarizationOutput:
    try:
        output = _DiarizationOutput.model_validate_json(unwrap_json_code_fence(content))
    except ValidationError as error:
        raise ModelOutputError("Omni diarization returned invalid structured output") from error
    if any(segment.end_ms > duration_ms for segment in output.segments):
        raise ModelOutputError("Omni diarization segment exceeds the clip duration")
    return output


def _parse_active_speakers(content: str) -> _ActiveSpeakerOutput:
    try:
        return _ActiveSpeakerOutput.model_validate_json(unwrap_json_code_fence(content))
    except ValidationError as error:
        raise ModelOutputError("active-speaker model returned invalid structured output") from error


def _read_bounded_media(media_path: Path, maximum_media_bytes: int) -> tuple[Path, bytes]:
    path = media_path.resolve(strict=True)
    if path.stat().st_size > maximum_media_bytes:
        raise ValueError("media exceeds the edge diarization request limit")
    return path, path.read_bytes()


def _annotated_face_video(
    media_path: Path,
    faces: tuple[IdentityObservationInput, ...],
    maximum_media_bytes: int,
    audio_path: Path | None = None,
) -> tuple[Path, bytes]:
    path = media_path.resolve(strict=True)
    sidecar = audio_path.resolve(strict=True) if audio_path is not None else None
    labels = {
        identity_id: f"F{index}"
        for index, identity_id in enumerate(dict.fromkeys(item.identity_id for item in faces))
    }
    filters: list[str] = []
    for face in faces:
        if face.visual_bbox_xyxy is None:
            continue
        left, top, right, bottom = face.visual_bbox_xyxy
        enabled = f"between(t\\,{face.start_ms / 1_000:.3f}\\,{face.end_ms / 1_000:.3f})"
        filters.extend(
            (
                (
                    f"drawbox=x=iw*{left:.6f}:y=ih*{top:.6f}:"
                    f"w=iw*{right - left:.6f}:h=ih*{bottom - top:.6f}:"
                    f"color=yellow@0.9:t=6:enable='{enabled}'"
                ),
                (
                    f"drawtext=text={labels[face.identity_id]}:x=main_w*{left:.6f}:"
                    f"y=max(0\\,main_h*{top:.6f}-38):fontsize=32:fontcolor=white:"
                    f"box=1:boxcolor=black@0.7:enable='{enabled}'"
                ),
            )
        )
    if not filters and sidecar is None:
        return _read_bounded_media(path, maximum_media_bytes)
    filters.extend(("scale=w='min(960\\,iw)':h=-2", "fps=12"))
    sidecar_input = ("-i", str(sidecar)) if sidecar is not None else ()
    audio_map = "1:a:0" if sidecar is not None else "0:a:0?"
    shortest = ("-shortest",) if sidecar is not None else ()
    with tempfile.NamedTemporaryFile(suffix=".mp4") as annotated:
        try:
            subprocess.run(
                (
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(path),
                    *sidecar_input,
                    "-vf",
                    ",".join(filters),
                    "-map",
                    "0:v:0",
                    "-map",
                    audio_map,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "28",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "64k",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    *shortest,
                    "-y",
                    annotated.name,
                ),
                check=True,
                capture_output=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ModelOutputError("FFmpeg could not annotate face anchors") from error
        return _read_bounded_media(Path(annotated.name), maximum_media_bytes)
