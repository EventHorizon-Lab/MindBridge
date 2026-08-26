"""Omni speaker-turn segmentation for the optional edge identity pipeline."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import re
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from importlib.util import find_spec
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
FUNASR_SENSEVOICE_MODEL_ID = "iic/SenseVoiceSmall"
FUNASR_NANO_MODEL_ID = "FunAudioLLM/Fun-ASR-Nano-2512"
FUNASR_STREAMING_ASR_MODEL_ID = (
    "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
)
FUNASR_VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
FUNASR_PUNCTUATION_MODEL_ID = "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
_PARALLEL_MODEL_MINIMUM_FREE_CUDA_BYTES = 8 * 1024 * 1024 * 1024
_SPEECH_SAMPLE_RATE = 16_000
# Upstream clusters CAM++ chunks at this threshold in both its own server and AutoModel.
_SPEAKER_CLUSTER_MERGE_THRESHOLD = 0.78
# Spans this short carry too few CAM++ chunks to place a speaker, so upstream drops them
# (`len(seg_audio) > sr * 0.3`, exclusive -- matched here so the two agree at the boundary).
_MINIMUM_TRANSCRIBED_SPAN_MS = 300
# Model special tokens: SenseVoice tags language, emotion and event inline
# (`<|zh|><|NEUTRAL|><|Speech|><|woitn|>`) and upstream strips them at the edge of its own
# server rather than inside the model, so every consumer has to do it. Left in, they reach
# the claim text as if someone had said them.
_MODEL_SPECIAL_TOKEN = re.compile(r"<\|[^|]*\|>")
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


class _Waveform(Protocol):
    """A decoded waveform, narrowed to the one thing the speech path does with it."""

    def __getitem__(self, span: slice) -> object: ...


class _FunASRNanoEngine(Protocol):
    def generate(
        self,
        *,
        inputs: list[object],
        **kwargs: object,
    ) -> list[dict[str, object]]: ...


@dataclass(frozen=True, slots=True)
class SpeechAnalysis:
    """Timed speech plus local speaker centroids, whatever backend produced them."""

    segments: tuple[SpeechSegment, ...]
    speaker_embeddings: tuple[SpeakerEmbeddingSample, ...]


class SpeechAnalyzer(Protocol):
    """The whole speech contract the identity path needs, with no backend in it.

    Both FunASR backends normalize to this: the portable `AutoModel` composition and the
    Fun-ASR-Nano vLLM engine. A third one only has to return timed spans and the speaker
    centroids those spans belong to.
    """

    async def analyze_file(self, media_path: Path) -> SpeechAnalysis: ...


@dataclass(frozen=True, slots=True)
class FunASRRecipe:
    """One FunASR model plus the models it needs composed around it.

    A FunASR model id alone does not say whether the checkpoint predicts timestamps, whether
    it punctuates its own output, or how upstream will then choose to segment speakers -- and
    those three answers differ per model while deciding whether `SpeechAnalysis` can be filled
    at all. Naming the composition is what makes a swap a configuration change instead of a
    silent downgrade, so a model MindBridge has not measured has to declare one.
    """

    model_id: str
    vad_model: str = FUNASR_VAD_MODEL_ID
    speaker_model: str = CAMPPLUS_MODEL.model_id
    punctuation_model: str | None = None
    vad_max_single_segment_ms: int | None = None
    trust_remote_code: bool = False
    # Upstream resolves an unset revision to "master", so a `trust_remote_code` model runs
    # whatever code the repository holds at load time. Pin this for a deployment that has
    # measured a specific checkpoint.
    revision: str | None = None
    # Whether `FunASRNanoVLLMPipeline` can serve these weights. Declared rather than inferred
    # from the model id, because whether a checkpoint is Fun-ASR-Nano's architecture is not
    # derivable from its name -- a local conversion or a fork of those weights is servable and
    # a Paraformer checkpoint is not, and both are just strings.
    vllm_servable: bool = False

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("FunASR recipe needs a model id")
        # Refuse here rather than after loading several GiB of weights and reading an empty
        # result: without VAD upstream never runs `inference_with_vad`, so there is no
        # `sentence_info` to time speech by, and without a speaker model there is no centroid,
        # which is the only thing a voiceprint can be matched against.
        if not self.vad_model.strip() or not self.speaker_model.strip():
            raise ValueError(
                "MindBridge speech analysis needs a VAD model for timed spans and a speaker "
                "model for voiceprint centroids"
            )
        if self.vad_max_single_segment_ms is not None and self.vad_max_single_segment_ms <= 0:
            raise ValueError("FunASR VAD segment ceiling must be positive")

    def auto_model_arguments(self) -> dict[str, object]:
        """Spell the recipe as upstream `AutoModel` keywords."""
        arguments: dict[str, object] = {
            "model": self.model_id,
            "vad_model": self.vad_model,
            "spk_model": self.speaker_model,
        }
        if self.punctuation_model is not None:
            arguments["punc_model"] = self.punctuation_model
        if self.vad_max_single_segment_ms is not None:
            arguments["vad_kwargs"] = {"max_single_segment_time": self.vad_max_single_segment_ms}
        if self.trust_remote_code:
            arguments["trust_remote_code"] = True
        if self.revision is not None:
            arguments["model_revision"] = self.revision
        return arguments


DEFAULT_FUNASR_RECIPE = "fun-asr-nano"

FUNASR_RECIPES: Mapping[str, FunASRRecipe] = {
    # The default. Fun-ASR-Nano punctuates its own output and emits CTC timestamps, so upstream
    # skips the punctuation model on its own (`punc_model is not None and "timestamps" not in
    # result`) and converts the dict timestamps to the list shape the rest of the pipeline
    # reads. Because punctuation never runs, `spk_mode` degrades to `vad_segment`: the text
    # carries the model's own punctuation, but the turn boundaries come from VAD, so the 30s
    # ceiling is what bounds a turn. `FunASRNanoVLLMPipeline` serves these same weights with
    # batched decoding and tighter bounds when a device has the CUDA headroom for vLLM.
    "fun-asr-nano": FunASRRecipe(
        model_id=FUNASR_NANO_MODEL_ID,
        vad_max_single_segment_ms=30_000,
        trust_remote_code=True,
        vllm_servable=True,
    ),
    # Character timestamps and a punctuation model: the only composition here that reaches
    # upstream's `punc_segment` diarization, so turns break on punctuation rather than on VAD.
    # Tuned for Mandarin -- measured on an English corpus this checkpoint transcribes the
    # right words with no word boundaries at all ("hellorobotwhathaveyouprepared..."), which
    # is recoverable but plainly not what the speech contained. Language is a property of
    # where the device sits, not of the product.
    "paraformer": FunASRRecipe(
        model_id=FUNASR_ASR_MODEL_ID,
        punctuation_model=FUNASR_PUNCTUATION_MODEL_ID,
    ),
    # SenseVoice predicts no timestamps, so upstream logs a warning and degrades to
    # `vad_segment` diarization; punctuation cannot be aligned to anything and would only
    # cost a model load, so VAD spans become the turns and the 30s ceiling bounds them.
    "sensevoice": FunASRRecipe(
        model_id=FUNASR_SENSEVOICE_MODEL_ID,
        vad_max_single_segment_ms=30_000,
    ),
}


def resolve_funasr_recipe(recipe: FunASRRecipe | str) -> FunASRRecipe:
    """Accept a measured recipe name, or a declared one for a model we have not measured."""
    if isinstance(recipe, FunASRRecipe):
        return recipe
    try:
        return FUNASR_RECIPES[recipe]
    except KeyError as error:
        raise ValueError(
            f"unknown FunASR recipe {recipe!r}; pass one of "
            f"{', '.join(sorted(FUNASR_RECIPES))} or a FunASRRecipe declaring the composition"
        ) from error


_ACTIVE_SPEAKER_SCHEMA = output_schema("active_speaker_match", _ActiveSpeakerOutput)

_SPEECH_SEGMENT_SCHEMA = output_schema("speech_segmentation", _DiarizationOutput)


class FunASRAutoModelPipeline:
    """Run one FunASR `AutoModel` composition: VAD, ASR, punctuation, diarization, centroids.

    This is the portable backend, and it is the generic one: upstream's `AutoModel` already
    normalizes across checkpoints -- it converts Fun-ASR-Nano's dict timestamps to the list
    shape, skips punctuation for models that punctuate themselves, and degrades speaker
    segmentation from `punc_segment` to `vad_segment` for models that predict no timestamps.
    What it does not do is choose the composition, which is what `FunASRRecipe` carries.
    """

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
    def load(
        cls,
        *,
        device: str | None = None,
        recipe: FunASRRecipe | str = DEFAULT_FUNASR_RECIPE,
    ) -> FunASRAutoModelPipeline:
        """Load a named recipe, or a declared one, on CUDA when available."""
        selected_recipe = resolve_funasr_recipe(recipe)
        try:
            funasr = __import__("funasr")
        except ImportError as error:
            raise ModelUnavailableError(
                "install the edge extra on Linux/Windows x86_64 or Apple Silicon macOS, "
                "or provide the platform FunASR runtime"
            ) from error
        selected_device = select_torch_device(device)
        pipeline = funasr.AutoModel(
            **selected_recipe.auto_model_arguments(),
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
        # Silence is not the same shape in every model: Paraformer returns an empty string,
        # while SenseVoice still tags the span (`<|zh|><|NEUTRAL|><|Speech|><|woitn|>`). Both
        # mean nobody spoke, so both have to reach the empty result rather than the error below.
        if not _MODEL_SPECIAL_TOKEN.sub("", text).strip():
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


class FunASRNanoVLLMPipeline:
    """Serve Fun-ASR-Nano on vLLM, then add the speaker evidence the engine cannot produce.

    The engine only transcribes: it takes audio arrays and returns text with CTC character
    timestamps, with no VAD, no speaker labels and no centroids. So this composes what
    upstream's own server composes around it -- FSMN-VAD to find the spans to batch, and CAM++
    to cluster them -- with one deliberate difference. Upstream's `attach_speaker_labels`
    computes the CAM++ embeddings, keeps only the `SPK{n}` labels and drops the vectors;
    MindBridge's voiceprint memory is built on exactly those per-speaker centroids, so this
    asks `postprocess` for the centers as well. Everything after that is the shared
    normalization, byte for byte the same functions the `AutoModel` backend goes through.
    """

    def __init__(
        self,
        engine: _FunASRNanoEngine,
        vad_pipeline: _FunASRPipeline,
        speaker_pipeline: _FunASRPipeline,
        *,
        device: str,
        diarization_confidence: float = 0.8,
        max_new_tokens: int = 500,
    ) -> None:
        if not 0.0 <= diarization_confidence <= 1.0:
            raise ValueError("diarization confidence must be between zero and one")
        if max_new_tokens <= 0:
            raise ValueError("Fun-ASR-Nano token budget must be positive")
        self._engine = engine
        self._vad_pipeline = vad_pipeline
        self._speaker_pipeline = speaker_pipeline
        self.device = device
        self._diarization_confidence = diarization_confidence
        self._max_new_tokens = max_new_tokens

    @classmethod
    def load(
        cls,
        *,
        device: str | None = None,
        model_id: str = FUNASR_NANO_MODEL_ID,
        revision: str | None = None,
        hub: str = "ms",
        vad_model: str = FUNASR_VAD_MODEL_ID,
        speaker_model: str = CAMPPLUS_MODEL.model_id,
        gpu_memory_utilization: float = 0.5,
        max_model_len: int = 4_096,
    ) -> FunASRNanoVLLMPipeline:
        """Load the vLLM engine plus the VAD and speaker models it has to be wrapped in."""
        # Upstream's `from_pretrained` forwards a revision to ModelScope's snapshot download and
        # drops it on the HuggingFace path. Dropping a pin quietly is how an unreviewed
        # checkpoint gets loaded under `trust_remote_code`, so refuse the combination instead.
        if revision is not None and hub not in {"ms", "modelscope"}:
            raise ValueError(
                "the Fun-ASR-Nano vLLM engine can only pin a revision on the ModelScope hub; "
                "upstream's HuggingFace path ignores it"
            )
        try:
            funasr = __import__("funasr")
            inference_vllm = __import__(
                "funasr.models.fun_asr_nano.inference_vllm",
                fromlist=["FunASRNanoVLLM"],
            )
        except ImportError as error:
            raise ModelUnavailableError(
                "install the edge extra plus vLLM, or use FunASRAutoModelPipeline with the "
                "'fun-asr-nano' recipe, which runs the same model on the portable runtime"
            ) from error
        selected_device = select_torch_device(device)
        if not selected_device.startswith("cuda"):
            raise ModelUnavailableError(
                "the Fun-ASR-Nano vLLM engine needs CUDA; use FunASRAutoModelPipeline with "
                "the 'fun-asr-nano' recipe on CPU-only devices"
            )
        engine = inference_vllm.FunASRNanoVLLM.from_pretrained(
            model=model_id,
            hub=hub,
            device=selected_device,
            dtype="bf16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            **({"revision": revision} if revision is not None else {}),
        )
        return cls(
            cast(_FunASRNanoEngine, engine),
            cast(
                _FunASRPipeline,
                funasr.AutoModel(
                    model=vad_model,
                    device=selected_device,
                    disable_update=True,
                    disable_pbar=True,
                ),
            ),
            cast(
                _FunASRPipeline,
                funasr.AutoModel(
                    model=speaker_model,
                    device=selected_device,
                    disable_update=True,
                    disable_pbar=True,
                ),
            ),
            device=selected_device,
        )

    async def analyze_file(self, media_path: Path) -> SpeechAnalysis:
        """Keep the edge loop responsive while local GPU inference runs."""
        return await asyncio.to_thread(self._analyze_file, media_path)

    def _analyze_file(self, media_path: Path) -> SpeechAnalysis:
        media_path = media_path.resolve(strict=True)
        audio = _load_speech_waveform(media_path)
        spans = self._speech_spans(audio)
        if not spans:
            return SpeechAnalysis(segments=(), speaker_embeddings=())
        results = self._engine.generate(
            inputs=[audio[_sample_index(start) : _sample_index(end)] for start, end in spans],
            max_new_tokens=self._max_new_tokens,
            # The engine runs in prompt-embeds mode, where any other value crashes the CUDA
            # kernel (upstream issue #2948). Passed explicitly so it reads as a constraint
            # rather than a default nobody chose.
            repetition_penalty=1.0,
        )
        if len(results) != len(spans):
            raise ModelOutputError("Fun-ASR-Nano returned an invalid transcription batch")
        sentences = _nano_sentences(results, spans)
        if not sentences:
            return SpeechAnalysis(segments=(), speaker_embeddings=())
        centroids = self._label_speakers(audio, sentences)
        return SpeechAnalysis(
            segments=_funasr_segments(sentences, confidence=self._diarization_confidence),
            speaker_embeddings=_funasr_speaker_embeddings(centroids),
        )

    def _speech_spans(self, audio: _Waveform) -> tuple[tuple[int, int], ...]:
        """Ask FSMN-VAD which millisecond spans to batch through the engine.

        Upstream's server falls back to the whole clip when VAD finds nothing, because a
        caller who posted a file asked for it to be transcribed. Here an empty VAD result is
        the answer: nobody spoke, and inventing a span over silence would enter the memory as
        speech.
        """
        output = self._vad_pipeline.generate(input=audio, fs=_SPEECH_SAMPLE_RATE)
        if len(output) != 1:
            raise ModelOutputError("FunASR VAD returned an invalid batch")
        value = output[0].get("value")
        if value is None:
            return ()
        if not isinstance(value, list) or len(value) > MAX_SPEECH_SEGMENTS:
            raise ModelOutputError("FunASR VAD returned invalid speech spans")
        spans = []
        for span in value:
            if (
                not isinstance(span, (list, tuple))
                or len(span) < 2
                or not all(
                    isinstance(edge, (int, float)) and not isinstance(edge, bool)
                    for edge in span[:2]
                )
            ):
                raise ModelOutputError("FunASR VAD returned an invalid speech span")
            start, end = round(span[0]), round(span[1])
            if end - start > _MINIMUM_TRANSCRIBED_SPAN_MS:
                spans.append((start, end))
        return tuple(spans)

    def _label_speakers(
        self,
        audio: _Waveform,
        sentences: list[dict[str, object]],
    ) -> object:
        """Cluster CAM++ chunks with upstream's own primitives and keep the centroids.

        `sentences` is mutated in place with the `spk` key, exactly as upstream's
        `distribute_spk` does inside `AutoModel`, so both backends hand the normalizer the
        same sentence shape.
        """
        torch = __import__("torch")
        numpy = __import__("numpy")
        campplus = __import__(
            "funasr.models.campplus.utils",
            fromlist=["distribute_spk", "postprocess", "sv_chunk"],
        )
        cluster_backend = __import__(
            "funasr.models.campplus.cluster_backend",
            fromlist=["ClusterBackend"],
        )
        # `sv_chunk` works in seconds and returns chunks in input order. That order is left
        # alone: `postprocess` pairs `segments[i]` with `labels[i]`, and the labels come back
        # in chunk order, so sorting the chunks without reordering the embeddings would
        # mislabel them. VAD spans are already chronological, so there is nothing to sort.
        chunks = campplus.sv_chunk(
            [
                [
                    cast(int, sentence["start"]) / 1_000,
                    cast(int, sentence["end"]) / 1_000,
                    audio[
                        _sample_index(cast(int, sentence["start"])) : _sample_index(
                            cast(int, sentence["end"])
                        )
                    ],
                ]
                for sentence in sentences
            ],
            fs=_SPEECH_SAMPLE_RATE,
        )
        if not chunks:
            return None
        embeddings = torch.cat(
            [
                result["spk_embedding"]
                for result in self._speaker_pipeline.generate(
                    input=[chunk[2] for chunk in chunks],
                    cache={},
                    is_final=True,
                )
            ],
            dim=0,
        )
        labels = numpy.asarray(
            cluster_backend.ClusterBackend(merge_thr=_SPEAKER_CLUSTER_MERGE_THRESHOLD).to(
                self.device
            )(embeddings.cpu(), oracle_num=None)
        )
        timeline, centroids = campplus.postprocess(
            chunks,
            None,
            labels,
            embeddings.detach().cpu().numpy(),
            return_spk_center=True,
        )
        campplus.distribute_spk(sentences, timeline)
        return centroids


SPEECH_ENGINES = ("automodel", "vllm")


def load_speech_analyzer(
    *,
    engine: str | None = None,
    device: str | None = None,
    recipe: FunASRRecipe | str = DEFAULT_FUNASR_RECIPE,
) -> SpeechAnalyzer:
    """Pick the inference engine for this device, or take the one named.

    Left unset this resolves from the device and the recipe together: a CUDA host with vLLM
    installed, asked for the Fun-ASR-Nano weights that engine serves, gets `vllm`; everything
    else gets `automodel`. Both fill the whole `SpeechAnalysis` contract, so within one model
    the choice is throughput and span precision rather than capability -- which is what makes
    it safe to decide automatically. Across models it is not, so the recipe constrains it: an
    engine is never allowed to quietly transcribe with a model nobody asked for.
    """
    selected_recipe = validate_speech_engine(engine, recipe)
    selected = (
        engine.strip().lower()
        if engine
        else _engine_for_environment(device=device, recipe=selected_recipe)
    )
    if selected == "vllm":
        return FunASRNanoVLLMPipeline.load(
            device=device,
            model_id=selected_recipe.model_id,
            revision=selected_recipe.revision,
        )
    return FunASRAutoModelPipeline.load(device=device, recipe=selected_recipe)


def validate_speech_engine(
    engine: str | None,
    recipe: FunASRRecipe | str = DEFAULT_FUNASR_RECIPE,
) -> FunASRRecipe:
    """Refuse an engine and recipe that cannot work together, and return the resolved recipe.

    Whether an engine can serve a recipe is a property of two strings -- no weights, no device,
    no GPU. So a deployment naming an impossible pair should hear about it while it is starting,
    not on the first clip that reaches a worker. Kept separate from `load_speech_analyzer` for
    exactly that: a caller that loads the pipeline lazily still wants the answer eagerly.
    """
    selected_recipe = resolve_funasr_recipe(recipe)
    selected = (engine or "").strip().lower()
    if selected and selected not in SPEECH_ENGINES:
        raise ValueError(
            f"unknown speech engine {selected!r}; pass one of {', '.join(SPEECH_ENGINES)}"
        )
    if selected == "vllm" and not selected_recipe.vllm_servable:
        raise ValueError(
            f"the vLLM engine cannot run {selected_recipe.model_id!r}: it implements "
            "Fun-ASR-Nano's architecture only. Use engine='automodel' for this model, or "
            "set vllm_servable on a recipe naming Fun-ASR-Nano weights"
        )
    return selected_recipe


def _engine_for_environment(*, device: str | None, recipe: FunASRRecipe) -> str:
    """CUDA with vLLM installed, running the weights it serves, goes to vLLM; else `AutoModel`.

    The recipe decides as much as the hardware does. This engine implements Fun-ASR-Nano's
    architecture only, so a host configured for Paraformer or SenseVoice gets `AutoModel`
    however much CUDA it has: choosing otherwise would transcribe with a different model, and
    a different language profile, than the deployment asked for. Silently substituting a model
    is the failure a recipe exists to prevent, and an automatic choice is the last place it
    should be reintroduced.

    vLLM also has to be actually importable, not merely wanted: it is a deliberate install, so
    treating its presence as the signal keeps a GPU host that never installed it on the
    portable path instead of failing at load.
    """
    # `select_torch_device` raises when an accelerator is demanded and missing, which is the
    # module's standing rule: a silent CPU fallback turns capacity into a latency mystery.
    if (
        recipe.vllm_servable
        and select_torch_device(device).startswith("cuda")
        and find_spec("vllm") is not None
    ):
        return "vllm"
    return "automodel"


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
    def load(
        cls,
        *,
        device: str | None = None,
        model_id: str = FUNASR_STREAMING_ASR_MODEL_ID,
    ) -> FunASRStreamingTranscriber:
        """Load a causal checkpoint only when live ASR is needed.

        `model_id` has to name an online/streaming FunASR checkpoint. Streaming is not a
        capability every FunASR model has -- SenseVoice and Fun-ASR-Nano have no causal
        variant -- so this stays a separate load rather than something a recipe can turn on.
        """
        try:
            funasr = __import__("funasr")
        except ImportError as error:
            raise ModelUnavailableError(
                "install the edge extra on Linux/Windows x86_64 or Apple Silicon macOS, "
                "or provide the platform FunASR runtime"
            ) from error
        selected_device = select_torch_device(device)
        pipeline = funasr.AutoModel(
            model=model_id,
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
    speech_pipeline: SpeechAnalyzer,
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


def _funasr_sentence_text(value: object) -> str | None:
    """Accept either FunASR sentence shape, because it emits both.

    `sentence_info[i]["text"]` is a plain string on some decode paths and a per-token list on
    others -- `['哦', '哦', '好', ...]` -- and which one you get varies with the model and the
    clip, not with the call. Rejecting the list shape discarded every sentence in the clip:
    measured on this corpus, **25 of 40 clips (62.5%) produced no transcript at all** for that
    reason alone, which is most of the speech a conversational benchmark depends on.

    Tokens are joined without a separator because these models tokenise Chinese per character;
    a token that already carries its own spacing keeps it.
    """
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (list, tuple)) and value:
        if not all(isinstance(token, str) for token in value):
            return None
        joined = "".join(value).strip()
        return joined or None
    return None


def _funasr_segments(value: object, *, confidence: float) -> tuple[SpeechSegment, ...]:
    if not isinstance(value, list) or len(value) > MAX_SPEECH_SEGMENTS:
        raise ModelOutputError("FunASR returned invalid sentence-level diarization")
    segments = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ModelOutputError("FunASR returned an invalid sentence")
        start, end = item.get("start"), item.get("end")
        transcript = _funasr_sentence_text(item.get("text") or item.get("sentence"))
        speaker = item.get("spk")
        if (
            not isinstance(start, (int, float))
            or isinstance(start, bool)
            or not isinstance(end, (int, float))
            or isinstance(end, bool)
            or transcript is None
            or (speaker is not None and not isinstance(speaker, (str, int)))
        ):
            raise ModelOutputError("FunASR returned an invalid sentence")
        # A sentence that is nothing but model special tokens is a VAD span the model heard no
        # speech in, which SenseVoice reports constantly. Dropping the span is right; treating
        # it as a malformed result would lose every other sentence in the clip with it, and a
        # missing or blank `text` still does raise above.
        transcript = _MODEL_SPECIAL_TOKEN.sub("", transcript).strip()
        if not transcript:
            continue
        start_ms, end_ms = round(start), round(end)
        try:
            segments.append(
                SpeechSegment(
                    sample_id=f"funasr-{start_ms:012d}-{end_ms:012d}-{index:03d}",
                    start_ms=start_ms,
                    end_ms=end_ms,
                    confidence=confidence,
                    transcript=transcript,
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


def _sample_index(milliseconds: int) -> int:
    return int(milliseconds * _SPEECH_SAMPLE_RATE / 1_000)


def _load_speech_waveform(media_path: Path) -> _Waveform:
    """Decode to 16 kHz mono through FunASR's own loader.

    Reusing it keeps the two backends on identical preprocessing -- the `AutoModel` path hands
    the file to this same function internally -- and inherits its torchaudio/soundfile/ffmpeg
    fallback chain, so no decoder becomes a MindBridge dependency.
    """
    load_utils = __import__("funasr.utils.load_utils", fromlist=["load_audio_text_image_video"])
    waveform = load_utils.load_audio_text_image_video(str(media_path), fs=_SPEECH_SAMPLE_RATE)
    return cast(_Waveform, waveform.detach().cpu().numpy())


def _nano_sentences(
    results: list[dict[str, object]],
    spans: tuple[tuple[int, int], ...],
) -> list[dict[str, object]]:
    """Put the engine's per-span text back on the clip's timeline.

    Fun-ASR-Nano is handed one VAD span at a time, so its CTC timestamps are relative to that
    span. Where they survived the forced alignment they are preferred over the VAD edges,
    which are padded: they are what the model actually heard, and they keep this backend's
    spans as tight as the `AutoModel` path's.
    """
    sentences: list[dict[str, object]] = []
    for result, (span_start, span_end) in zip(results, spans, strict=True):
        text = result.get("text")
        if not isinstance(text, str):
            raise ModelOutputError("Fun-ASR-Nano returned invalid transcription text")
        if not _MODEL_SPECIAL_TOKEN.sub("", text).strip():
            continue
        start, end = span_start, span_end
        timestamps = result.get("timestamps")
        if isinstance(timestamps, list) and timestamps:
            aligned = _nano_span(timestamps, span_start, span_end)
            if aligned is not None:
                start, end = aligned
        sentences.append({"start": start, "end": end, "text": text})
    return sentences


def _nano_span(timestamps: list[object], span_start: int, span_end: int) -> tuple[int, int] | None:
    """Clamp the CTC character alignment into the VAD span it was measured inside."""
    edges = []
    for item in timestamps:
        if not isinstance(item, dict):
            return None
        start, end = item.get("start_time"), item.get("end_time")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            return None
        edges.append((float(start), float(end)))
    if not edges:
        return None
    start = span_start + round(min(edge[0] for edge in edges) * 1_000)
    end = span_start + round(max(edge[1] for edge in edges) * 1_000)
    start = min(max(start, span_start), span_end)
    end = min(max(end, start), span_end)
    return (start, end) if end > start else (span_start, span_end)


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
