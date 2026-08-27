"""FunASR speech analysis with Fun-ASR-Nano as the default recipe."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from itertools import pairwise
from threading import RLock
from typing import Protocol, cast

from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.models.base import SpeakerEmbedding, SpeechAnalysis, SpeechTurn
from mindbridge.types import AssetRef, Modality

DEFAULT_FUNASR_MODEL_ID = "FunAudioLLM/Fun-ASR-Nano-2512"
DEFAULT_FUNASR_MODEL_REVISION = "05201c46f1c38592b1567f857c0d56eab3d0d8ef"
DEFAULT_FUNASR_VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
DEFAULT_FUNASR_VAD_REVISION = "f9a8b8274674755d925277e27063869038d41515"
DEFAULT_FUNASR_SPEAKER_MODEL_ID = "iic/speech_campplus_sv_zh-cn_16k-common"
DEFAULT_FUNASR_SPEAKER_REVISION = "a045b2afcaa9c3049c98a9215a2bc274407ab237"
_MAX_TURNS = 10_000
_MAX_TURN_CHARACTERS = 4_096
_MINIMUM_SPAN_MS = 300
_SAMPLE_RATE = 16_000
_SPEAKER_CLUSTER_MERGE_THRESHOLD = 0.78
_SPECIAL_TOKEN = re.compile(r"<\|[^|]*\|>")


class _Pipeline(Protocol):
    def generate(self, **kwargs: object) -> list[dict[str, object]]: ...


class _Waveform(Protocol):
    def __getitem__(self, span: slice) -> object: ...


class _VLLMEngine(Protocol):
    def generate(
        self,
        *,
        inputs: list[object],
        **kwargs: object,
    ) -> list[dict[str, object]]: ...


@dataclass(slots=True)
class _VLLMPipeline:
    engine: _VLLMEngine
    vad: _Pipeline
    speaker: _Pipeline
    device: str

    def close(self) -> None:
        for component in (self.engine, self.vad, self.speaker):
            close = getattr(component, "close", None)
            if callable(close):
                close()


@dataclass(frozen=True, slots=True)
class FunASRRecipe:
    """One ASR model and the FunASR components composed around it."""

    model_id: str
    vad_model: str
    speaker_model: str
    model_revision: str | None = None
    vad_revision: str | None = None
    speaker_revision: str | None = None
    punctuation_model: str | None = None
    punctuation_revision: str | None = None
    vad_max_single_segment_ms: int | None = None
    hub: str = "ms"
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        required = (self.model_id, self.vad_model, self.speaker_model, self.hub)
        if any(not isinstance(value, str) or not value.strip() for value in required):
            raise ValidationError("FunASR recipe model and hub values must not be blank")
        optional = (
            self.model_revision,
            self.vad_revision,
            self.speaker_revision,
            self.punctuation_model,
            self.punctuation_revision,
        )
        if any(
            value is not None and (not isinstance(value, str) or not value.strip())
            for value in optional
        ):
            raise ValidationError("FunASR optional recipe values must not be blank")
        if self.vad_max_single_segment_ms is not None and (
            isinstance(self.vad_max_single_segment_ms, bool) or self.vad_max_single_segment_ms <= 0
        ):
            raise ValidationError("FunASR VAD segment ceiling must be positive")

    def auto_model_arguments(self) -> dict[str, object]:
        """Return the standard FunASR AutoModel composition arguments."""
        values: dict[str, object] = {
            "model": self.model_id,
            "vad_model": self.vad_model,
            "spk_model": self.speaker_model,
            "hub": self.hub,
        }
        optional = {
            "model_revision": self.model_revision,
            "vad_model_revision": self.vad_revision,
            "spk_model_revision": self.speaker_revision,
            "punc_model": self.punctuation_model,
            "punc_model_revision": self.punctuation_revision,
        }
        values.update((key, value) for key, value in optional.items() if value is not None)
        if self.vad_max_single_segment_ms is not None:
            values["vad_kwargs"] = {"max_single_segment_time": self.vad_max_single_segment_ms}
        if self.trust_remote_code:
            values["trust_remote_code"] = True
        return values


DEFAULT_FUNASR_RECIPE = FunASRRecipe(
    model_id=DEFAULT_FUNASR_MODEL_ID,
    model_revision=DEFAULT_FUNASR_MODEL_REVISION,
    vad_model=DEFAULT_FUNASR_VAD_MODEL_ID,
    vad_revision=DEFAULT_FUNASR_VAD_REVISION,
    speaker_model=DEFAULT_FUNASR_SPEAKER_MODEL_ID,
    speaker_revision=DEFAULT_FUNASR_SPEAKER_REVISION,
    vad_max_single_segment_ms=30_000,
)


class FunASRTranscriber:
    """Lazily run FunASR-Nano with portable AutoModel or batched vLLM decoding."""

    def __init__(
        self,
        recipe: FunASRRecipe = DEFAULT_FUNASR_RECIPE,
        *,
        device: str = "auto",
        engine: str = "automodel",
        gpu_memory_utilization: float = 0.5,
        max_model_len: int = 4_096,
        max_new_tokens: int = 500,
        tensor_parallel_size: int = 1,
        vllm_dtype: str = "bf16",
    ) -> None:
        if not isinstance(recipe, FunASRRecipe):
            raise ValidationError("recipe must be a FunASRRecipe value")
        normalized_device = device.strip().lower() if isinstance(device, str) else ""
        if (
            normalized_device not in {"auto", "cpu"}
            and re.fullmatch(r"cuda(?::\d+)?", normalized_device) is None
        ):
            raise ValidationError("device must be auto, cpu, cuda, or cuda:<index>")
        normalized_engine = engine.strip().lower() if isinstance(engine, str) else ""
        if normalized_engine not in {"automodel", "vllm"}:
            raise ValidationError("engine must be automodel or vllm")
        normalized_dtype = vllm_dtype.strip().lower() if isinstance(vllm_dtype, str) else ""
        if normalized_dtype not in {"bf16", "fp16", "fp32"}:
            raise ValidationError("vllm_dtype must be bf16, fp16, or fp32")
        if (
            isinstance(gpu_memory_utilization, bool)
            or not isinstance(gpu_memory_utilization, int | float)
            or not math.isfinite(float(gpu_memory_utilization))
            or not 0.0 < gpu_memory_utilization <= 1.0
        ):
            raise ValidationError(
                "gpu_memory_utilization must be greater than zero and at most one"
            )
        for value, name in (
            (max_model_len, "max_model_len"),
            (max_new_tokens, "max_new_tokens"),
            (tensor_parallel_size, "tensor_parallel_size"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(f"{name} must be a positive integer")
        self._recipe = recipe
        self._device = normalized_device
        self._engine = normalized_engine
        self._gpu_memory_utilization = float(gpu_memory_utilization)
        self._max_model_len = max_model_len
        self._max_new_tokens = max_new_tokens
        self._tensor_parallel_size = tensor_parallel_size
        self._vllm_dtype = normalized_dtype
        self._runtime_selection: tuple[str, str] | None = None
        self._pipeline: _Pipeline | _VLLMPipeline | None = None
        # ponytail: FunASR inference is serialized per loaded model; relax only after its
        # AutoModel composition demonstrates thread safety and concurrent throughput gains.
        self._lock = RLock()
        self._closed = False

    @property
    def capabilities(self) -> frozenset[Modality]:
        return frozenset({Modality.AUDIO, Modality.VIDEO})

    @property
    def model_id(self) -> str:
        return self._recipe.model_id

    @property
    def space_id(self) -> str:
        engine, _device = self._runtime()
        configuration: dict[str, object] = {
            "engine": engine,
            "recipe": self._recipe.auto_model_arguments(),
        }
        if engine == "vllm":
            configuration.update(
                max_model_len=self._max_model_len,
                max_new_tokens=self._max_new_tokens,
                vllm_dtype=self._vllm_dtype,
            )
        payload = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(f"funasr-analysis-v2:{payload}".encode()).hexdigest()[:16]
        return f"{self.model_id}:speech:{digest}"

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]:
        if isinstance(assets, (str, bytes)):
            raise ValidationError("assets must contain resolved audio or video AssetRef values")
        batch = tuple(assets)
        for asset in batch:
            if (
                not isinstance(asset, AssetRef)
                or not asset.is_resolved
                or asset.modality not in self.capabilities
            ):
                raise ValidationError("assets must contain resolved audio or video AssetRef values")
        with self._lock:
            if self._closed:
                raise ModelError("FunASR transcriber is closed")
            pipeline = self._load()
            if isinstance(pipeline, _VLLMPipeline):
                return self._analyze_vllm(pipeline, batch)
            return tuple(self._analyze_one(pipeline, asset) for asset in batch)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pipeline, self._pipeline = self._pipeline, None
            close = getattr(pipeline, "close", None)
            if callable(close):
                close()

    def _load(self) -> _Pipeline | _VLLMPipeline:
        if self._pipeline is not None:
            return self._pipeline
        engine, device = self._runtime()
        if engine == "vllm":
            self._pipeline = self._load_vllm(device)
            return self._pipeline
        try:
            funasr = import_module("funasr")
        except ImportError:
            raise ModelError(
                "FunASR is required for local speech; install mindbridge[local]"
            ) from None
        try:
            pipeline = funasr.AutoModel(
                **self._recipe.auto_model_arguments(),
                device=device,
                disable_update=True,
                disable_pbar=True,
            )
        except Exception as error:
            raise ModelError("failed to load the FunASR speech recipe") from error
        self._pipeline = cast(_Pipeline, pipeline)
        return self._pipeline

    def _load_vllm(self, device: str) -> _VLLMPipeline:
        if self._recipe.model_revision is not None and self._recipe.hub not in {"ms", "modelscope"}:
            raise ValidationError(
                "FunASR vLLM can only enforce model_revision with the ModelScope hub"
            )
        try:
            funasr = import_module("funasr")
            vllm_module = import_module("funasr.auto.auto_model_vllm")
        except ImportError:
            raise ModelError(
                "vLLM is required for this speech engine; install a driver-compatible vLLM "
                "before mindbridge[local,vllm]"
            ) from None
        engine_arguments: dict[str, object] = {
            "model": self._recipe.model_id,
            "hub": self._recipe.hub,
            "device": device,
            "dtype": self._vllm_dtype,
            "gpu_memory_utilization": self._gpu_memory_utilization,
            "max_model_len": self._max_model_len,
            "tensor_parallel_size": self._tensor_parallel_size,
        }
        if self._recipe.model_revision is not None:
            engine_arguments["revision"] = self._recipe.model_revision
        vad_arguments: dict[str, object] = {
            "model": self._recipe.vad_model,
            "hub": self._recipe.hub,
            "device": device,
            "disable_update": True,
            "disable_pbar": True,
        }
        if self._recipe.vad_revision is not None:
            vad_arguments["model_revision"] = self._recipe.vad_revision
        if self._recipe.vad_max_single_segment_ms is not None:
            vad_arguments["max_single_segment_time"] = self._recipe.vad_max_single_segment_ms
        speaker_arguments: dict[str, object] = {
            "model": self._recipe.speaker_model,
            "hub": self._recipe.hub,
            "device": device,
            "disable_update": True,
            "disable_pbar": True,
        }
        if self._recipe.speaker_revision is not None:
            speaker_arguments["model_revision"] = self._recipe.speaker_revision
        try:
            return _VLLMPipeline(
                engine=cast(_VLLMEngine, vllm_module.AutoModelVLLM(**engine_arguments)),
                vad=cast(_Pipeline, funasr.AutoModel(**vad_arguments)),
                speaker=cast(_Pipeline, funasr.AutoModel(**speaker_arguments)),
                device=device,
            )
        except Exception as error:
            raise ModelError("failed to load the FunASR vLLM speech recipe") from error

    def _runtime(self) -> tuple[str, str]:
        if self._runtime_selection is not None:
            return self._runtime_selection
        device = self._selected_device()
        engine = self._engine
        if engine == "vllm" and not device.startswith("cuda"):
            raise ModelError("the FunASR vLLM engine requires a CUDA device")
        self._runtime_selection = (engine, device)
        return self._runtime_selection

    def _analyze_vllm(
        self,
        pipeline: _VLLMPipeline,
        assets: tuple[AssetRef, ...],
    ) -> tuple[SpeechAnalysis, ...]:
        try:
            waveforms = tuple(_load_waveform(asset) for asset in assets)
            spans_by_asset = tuple(_speech_spans(pipeline.vad, audio) for audio in waveforms)
            inputs = [
                audio[_sample_index(start) : _sample_index(end)]
                for audio, spans in zip(waveforms, spans_by_asset, strict=True)
                for start, end in spans
            ]
            if not inputs:
                return tuple(SpeechAnalysis(turns=(), speakers=()) for _asset in assets)
            results = pipeline.engine.generate(
                inputs=inputs,
                max_new_tokens=self._max_new_tokens,
                # FunASR uses prompt embeddings; non-neutral repetition penalties can trigger
                # an out-of-bounds CUDA assertion because there are no prompt token IDs.
                repetition_penalty=1.0,
            )
            if not isinstance(results, list) or len(results) != len(inputs):
                raise ModelError("FunASR vLLM returned an invalid transcription batch")
            analyses = []
            offset = 0
            for audio, spans in zip(waveforms, spans_by_asset, strict=True):
                sentences = _nano_sentences(results[offset : offset + len(spans)], spans)
                offset += len(spans)
                if not sentences:
                    analyses.append(SpeechAnalysis(turns=(), speakers=()))
                    continue
                centroids = _label_speakers(pipeline, audio, sentences)
                turns = _turns(sentences)
                speakers = _speakers(centroids)
                labels = {turn.speaker_label for turn in turns if turn.speaker_label is not None}
                if labels - {speaker.speaker_label for speaker in speakers}:
                    raise ModelError("FunASR returned a speaker label without a CAM++ centroid")
                analyses.append(SpeechAnalysis(turns=turns, speakers=speakers))
            return tuple(analyses)
        except (ModelError, ValidationError):
            raise
        except OSError:
            raise ValidationError("speech asset path does not exist") from None
        except Exception as error:
            raise ModelError("FunASR vLLM failed to analyze speech") from error

    def _selected_device(self) -> str:
        try:
            torch = import_module("torch")
            cuda_available = bool(torch.cuda.is_available())
        except (ImportError, RuntimeError):
            cuda_available = False
        if self._device == "auto":
            return "cuda" if cuda_available else "cpu"
        if self._device.startswith("cuda") and not cuda_available:
            raise ModelError("CUDA was requested for FunASR but is not available")
        return self._device

    @staticmethod
    def _analyze_one(pipeline: _Pipeline, asset: AssetRef) -> SpeechAnalysis:
        path = asset.path
        if path is None:  # AssetRef.is_resolved was checked at the boundary.
            raise ValidationError("speech asset path is missing")
        try:
            output = pipeline.generate(
                input=str(path.resolve(strict=True)),
                batch_size_s=300,
                return_raw_text=True,
                sentence_timestamp=True,
                return_spk_res=True,
                return_spk_center=True,
                disable_pbar=True,
            )
        except OSError:
            raise ValidationError("speech asset path does not exist") from None
        except Exception as error:
            raise ModelError("FunASR failed to analyze speech") from error
        if not isinstance(output, list) or len(output) != 1 or not isinstance(output[0], dict):
            raise ModelError("FunASR returned an invalid transcription batch")
        result = output[0]
        text = result.get("text")
        if not isinstance(text, str):
            raise ModelError("FunASR returned invalid transcription text")
        if not _SPECIAL_TOKEN.sub("", text).strip():
            return SpeechAnalysis(turns=(), speakers=())
        turns = _turns(result.get("sentence_info"))
        if not turns:
            raise ModelError("FunASR returned text without timed speaker turns")
        speakers = _speakers(result.get("spk_embedding_center"))
        labels = {turn.speaker_label for turn in turns if turn.speaker_label is not None}
        if labels - {speaker.speaker_label for speaker in speakers}:
            raise ModelError("FunASR returned a speaker label without a CAM++ centroid")
        return SpeechAnalysis(turns=turns, speakers=speakers)


def _turns(value: object) -> tuple[SpeechTurn, ...]:
    if not isinstance(value, list) or len(value) > _MAX_TURNS:
        raise ModelError("FunASR returned invalid sentence-level diarization")
    turns = []
    for item in value:
        turn = _turn(item)
        if turn is not None:
            turns.append(turn)
    if any(current.start_ms < previous.start_ms for previous, current in pairwise(turns)):
        raise ModelError("FunASR speaker turns are not chronological")
    return tuple(turns)


def _load_waveform(asset: AssetRef) -> _Waveform:
    path = asset.path
    if path is None:
        raise ValidationError("speech asset path is missing")
    loader = import_module("funasr.utils.load_utils")
    waveform = loader.load_audio_text_image_video(
        str(path.resolve(strict=True)),
        fs=_SAMPLE_RATE,
    )
    detach = getattr(waveform, "detach", None)
    if callable(detach):
        waveform = detach().cpu().numpy()
    return cast(_Waveform, waveform)


def _speech_spans(pipeline: _Pipeline, audio: _Waveform) -> tuple[tuple[int, int], ...]:
    output = pipeline.generate(input=audio, fs=_SAMPLE_RATE)
    if not isinstance(output, list) or len(output) != 1 or not isinstance(output[0], dict):
        raise ModelError("FunASR VAD returned an invalid batch")
    value = output[0].get("value")
    if value is None:
        return ()
    if not isinstance(value, list) or len(value) > _MAX_TURNS:
        raise ModelError("FunASR VAD returned invalid speech spans")
    spans = []
    for span in value:
        if (
            not isinstance(span, (list, tuple))
            or len(span) < 2
            or any(
                isinstance(edge, bool)
                or not isinstance(edge, int | float)
                or not math.isfinite(float(edge))
                for edge in span[:2]
            )
        ):
            raise ModelError("FunASR VAD returned an invalid speech span")
        start, end = round(span[0]), round(span[1])
        if start < 0 or end <= start:
            raise ModelError("FunASR VAD returned an invalid speech span")
        if end - start > _MINIMUM_SPAN_MS:
            spans.append((start, end))
    if any(current[0] < previous[1] for previous, current in pairwise(spans)):
        raise ModelError("FunASR VAD speech spans overlap or are not chronological")
    return tuple(spans)


def _nano_sentences(
    results: list[dict[str, object]],
    spans: tuple[tuple[int, int], ...],
) -> list[dict[str, object]]:
    sentences = []
    for result, (start, end) in zip(results, spans, strict=True):
        if not isinstance(result, dict):
            raise ModelError("FunASR vLLM returned invalid transcription text")
        text = result.get("text")
        if not isinstance(text, str):
            raise ModelError("FunASR vLLM returned invalid transcription text")
        if _SPECIAL_TOKEN.sub("", text).strip():
            # Nano is decoded one VAD span at a time. VAD boundaries remain reliable even
            # when this checkpoint cannot align every generated character.
            sentences.append({"start": start, "end": end, "text": text})
    return sentences


def _label_speakers(
    pipeline: _VLLMPipeline,
    audio: _Waveform,
    sentences: list[dict[str, object]],
) -> object:
    torch = import_module("torch")
    numpy = import_module("numpy")
    campplus = import_module("funasr.models.campplus.utils")
    cluster_module = import_module("funasr.models.campplus.cluster_backend")
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
        fs=_SAMPLE_RATE,
    )
    if not chunks:
        raise ModelError("CAM++ could not create speaker chunks")
    outputs = pipeline.speaker.generate(
        input=[chunk[2] for chunk in chunks],
        cache={},
        is_final=True,
    )
    if len(outputs) != len(chunks) or any(
        not isinstance(output, dict) or "spk_embedding" not in output for output in outputs
    ):
        raise ModelError("CAM++ returned invalid speaker embeddings")
    embeddings = torch.cat([output["spk_embedding"] for output in outputs], dim=0)
    labels = numpy.asarray(
        cluster_module.ClusterBackend(merge_thr=_SPEAKER_CLUSTER_MERGE_THRESHOLD).to(
            pipeline.device
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


def _sample_index(milliseconds: int) -> int:
    return int(milliseconds * _SAMPLE_RATE / 1_000)


def _turn(item: object) -> SpeechTurn | None:
    if not isinstance(item, Mapping):
        raise ModelError("FunASR returned an invalid speaker turn")
    start, end = item.get("start"), item.get("end")
    raw_text = item.get("text")
    if raw_text is None:
        raw_text = item.get("sentence")
    if isinstance(raw_text, str) and not _SPECIAL_TOKEN.sub("", raw_text).strip():
        return None
    text = _sentence_text(raw_text)
    speaker = item.get("spk")
    if (
        not isinstance(start, int | float)
        or isinstance(start, bool)
        or not isinstance(end, int | float)
        or isinstance(end, bool)
        or text is None
        or (speaker is not None and not isinstance(speaker, str | int))
    ):
        raise ModelError("FunASR returned an invalid speaker turn")
    text = _SPECIAL_TOKEN.sub("", text).strip()
    if not text:
        return None
    if len(text) > _MAX_TURN_CHARACTERS:
        raise ModelError("FunASR speaker turn text is too long")
    start_ms, end_ms = round(start), round(end)
    if start_ms < 0 or end_ms <= start_ms:
        raise ModelError("FunASR returned an invalid speaker turn range")
    label = str(speaker).strip() if speaker is not None else None
    if label == "":
        raise ModelError("FunASR returned an invalid speaker label")
    return SpeechTurn(start_ms, end_ms, text, label)


def _sentence_text(value: object) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if (
        isinstance(value, (list, tuple))
        and value
        and all(isinstance(token, str) for token in value)
    ):
        return "".join(value).strip() or None
    return None


def _speakers(value: object) -> tuple[SpeakerEmbedding, ...]:
    if value is None:
        return ()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if not isinstance(value, list):
        raise ModelError("FunASR returned invalid speaker centroids")
    rows = [value] if value and all(isinstance(number, int | float) for number in value) else value
    if any(not isinstance(row, list) for row in rows):
        raise ModelError("FunASR returned invalid speaker centroids")
    speakers = []
    for index, row in enumerate(rows):
        if not row or any(
            isinstance(number, bool)
            or not isinstance(number, int | float)
            or not math.isfinite(float(number))
            for number in row
        ):
            raise ModelError("FunASR returned invalid speaker centroids")
        values = tuple(float(number) for number in row)
        magnitude = math.sqrt(math.fsum(number * number for number in values))
        if magnitude == 0.0:
            raise ModelError("FunASR returned a zero speaker centroid")
        speakers.append(
            SpeakerEmbedding(str(index), tuple(number / magnitude for number in values))
        )
    return tuple(speakers)
